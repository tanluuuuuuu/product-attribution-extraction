from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification
from typing import Dict
from transformers import RobertaForTokenClassification
import evaluate
from torch.optim import AdamW
from transformers import get_scheduler
from accelerate import Accelerator

from data_utils import assign_ner_tags_roberta, load_data
from model_utils import save_model, get_output_dir, setup_model
from evaluate_model import evaluate_model

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    ner_tags = [
        "MAT",
        "COLOR",
    ]
    processed_ner_tags = ['O']
    for tag in ner_tags:
        processed_ner_tags.extend([f"B-{tag}", f"I-{tag}"])
    ner_tags_2_number = {t: i for (i, t) in enumerate(processed_ner_tags)}
    number_2_ner_tags = {t: i for (t, i) in enumerate(ner_tags_2_number)}

    train_dataset = load_data("data/train.xlsx", tokenizer, ner_tags_2_number)
    test_dataset = load_data("data/test.xlsx", tokenizer, ner_tags_2_number)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=4,
    )
    test_dataloader = DataLoader(
        test_dataset, collate_fn=data_collator, batch_size=4
    )

    model = setup_model(ner_tags_2_number, number_2_ner_tags)
    metric = evaluate.load("seqeval")
    optimizer = AdamW(model.parameters(), lr=2e-5)
    num_train_epochs = 30
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    accelerator = Accelerator()
    model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader
    )

    output_dir = get_output_dir()
    best_f1_score = 0
    for epoch in range(num_train_epochs):
        # Training
        model.train()
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
        # Evaluate
        results = evaluate_model(
            model,
            test_dataloader,
            accelerator,
            metric,
            processed_ner_tags,
            epoch
        )

        # Save and upload
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        if accelerator.is_main_process and results[f"overall_f1"] > best_f1_score:
            save_model(output_dir, 'best_f1', tokenizer,
                       unwrapped_model, accelerator.save)
            best_f1_score = results[f"overall_f1"]
            print(
                f"Save best f1 model at epoch {epoch} with better f1 score {best_f1_score}")
        if (epoch + 1) % 100 == 0:
            save_model(
                output_dir, f'epoch_{epoch + 1}', tokenizer, unwrapped_model, accelerator.save)
            print(f"Save model at epoch {epoch + 1}")
            
            
            
            
