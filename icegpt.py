import os

all_paths = []
for root, dirs, files in os.walk("./processed_data"):
    for file in files:
        if file.endswith(".txt"):
            all_paths.append(os.path.join(root, file))


# load tokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./new_tokenizer")


tokenizer("Hvað er að frétta 😁 ?")


from datasets import load_dataset

portion = 1
all_paths = all_paths[: int(len(all_paths) * portion)]

train_paths = all_paths[: int(len(all_paths) * 0.8)]
test_paths = all_paths[int(len(all_paths) * 0.8) :]

dataset = load_dataset(
    "text", data_files={"train": train_paths, "test": test_paths}, streaming=True
).with_format("torch")

train_dataset = dataset["train"]
test_dataset = dataset["test"]


context_length = 512


def tokenize_function(examples):
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


tokanized_train_dataset = train_dataset.map(
    tokenize_function, batched=True, batch_size=100, remove_columns=["text"]
)
tokanized_test_dataset = test_dataset.map(
    tokenize_function, batched=True, batch_size=100, remove_columns=["text"]
)

# next(iter(tokanized_train_dataset))
# next(iter(tokanized_test_dataset))


tokanized_train_dataset, tokanized_test_dataset


print(next(iter(tokanized_train_dataset))["input_ids"].shape)


from transformers import GPT2Config, GPT2LMHeadModel

scale = 1

# Initializing a GPT2 configuration
configuration = GPT2Config(
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    n_embd=int(768 * scale),
    n_layer=int(12 * scale),
    n_head=int(12 * scale),
)

# Initializing a model from the configuration
model = GPT2LMHeadModel(configuration)

# Accessing the model configuration
configuration = model.config


model.config


model_size = sum(t.numel() for t in model.parameters())
print(f"ICE GPT-2 size: {model_size/1000**2:.1f}M parameters")


from transformers import DataCollatorForLanguageModeling

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, pad_to_multiple_of=None
)


import torch

torch.cuda.is_available()


# !nvidia-smi


from transformers import Trainer, TrainingArguments
from transformers import AdamW

optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)


args = TrainingArguments(
    output_dir="icebreaker",
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="steps",
    # eval_steps=5_000,
    eval_steps=2,
    logging_steps=2,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    weight_decay=0.1,
    # warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=100,
    max_steps=10,
    # fp16=True,
)

# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokanized_train_dataset,
    eval_dataset=tokanized_test_dataset,
    data_collator=data_collator,
    optimizers=(optimizer, None),
)

trainer.train()


model.save_pretrained("icebreaker")
tokenizer.save_pretrained("icebreaker")


from transformers import TextGenerationPipeline

# Load the trained model
model_path = "icebreaker/checkpoint-100"
model = GPT2LMHeadModel.from_pretrained(model_path)
model.eval()

# Create a text generation pipeline
pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer)

# Generate text
generated_text = pipeline(
    "<Fréttir> Í gærkvöldi",
    max_length=1000,
    num_return_sequences=1,
    do_sample=True,
    top_k=50,
    top_p=0.8,
    temperature=1.0,
)

# Print the generated text
print(generated_text[0]["generated_text"])


from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

df = pd.DataFrame(trainer.state.log_history)
df.to_csv("./icebreaker/log_history.csv")

training_loss_data = []
eval_loss_data = []
for i in range(len(df)):
    row = df.iloc[i]
    epoch, train_loss, eval_loss = row["epoch"], row["loss"], row["eval_loss"]

    if not np.isnan(eval_loss):
        eval_loss_data.append((epoch, eval_loss))

    if not np.isnan(train_loss):
        training_loss_data.append((epoch, train_loss))

# Extract x and y values for training and evaluation losses
training_epochs, training_losses = zip(*training_loss_data)
eval_epochs, eval_losses = zip(*eval_loss_data)

# Create a figure and axis
fig, ax = plt.subplots()

# Plot the training loss
ax.plot(training_epochs, training_losses, label="Training Loss")

# Plot the evaluation loss
ax.plot(eval_epochs, eval_losses, label="Evaluation Loss")

# Set labels and title
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.set_title("Training and Evaluation Loss")

# Add a legend
ax.legend()

# Display the plot
plt.show()
