from transformers import Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer

MODELS_DIR = '/home/georgios.vernikos/data-local/hf_models/'

def t5_trainer(output_dir, model, tokenizer, tokenized_dataset, task_metrics, batch_size, steps, max_steps, selection_metric,
               max_target_length, grad_acc_steps):
    training_args = Seq2SeqTrainingArguments(output_dir=output_dir, do_train=True, do_eval=True,
                                             evaluation_strategy='steps',
                                             per_device_train_batch_size=batch_size,
                                             per_device_eval_batch_size=batch_size,
                                             learning_rate=0.001,
                                             gradient_accumulation_steps=grad_acc_steps,
                                             max_steps=max_steps, logging_steps=steps, save_steps=steps,
                                             metric_for_best_model=selection_metric,
                                             eval_steps=steps, optim='adafactor',
                                             predict_with_generate=True, generation_max_length=max_target_length)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model,
        training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=task_metrics
    )

    trainer.train()
