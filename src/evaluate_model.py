import torch
from data_utils import postprocess
from sklearn.metrics import classification_report

def evaluate_model(
    model,
    test_dataloader,
    accelerator,
    metric,
    processed_ner_tags,
    epoch
):
    # Evaluation
    model.eval()
    y_true = []
    y_pred = []
    for batch in test_dataloader:
        
        with torch.no_grad():
            outputs = model(**batch)

        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]

        # Necessary to pad predictions and labels for being gathered
        predictions = accelerator.pad_across_processes(
            predictions, dim=1, pad_index=-100)
        labels = accelerator.pad_across_processes(
            labels, dim=1, pad_index=-100)

        predictions_gathered = accelerator.gather(predictions)
        labels_gathered = accelerator.gather(labels)
        
        y_true.extend(labels.ravel().to('cpu').numpy())
        y_pred.extend(predictions.ravel().to('cpu').numpy())

        true_predictions, true_labels = postprocess(
            predictions_gathered, labels_gathered, processed_ner_tags)
        metric.add_batch(predictions=true_predictions,
                            references=true_labels)

    results = metric.compute()
    print(
        f"epoch {epoch}:",
        {
            key: results[f"overall_{key}"]
            for key in ["precision", "recall", "f1", "accuracy"]
        },
    )
    
    report = classification_report(y_true, y_pred, labels=[1, 2, 3, 4], target_names=['B-MAT', 'I-MAT', 'B-COLOR', 'B-COLOR'], digits=4)
    print(report)
    return results