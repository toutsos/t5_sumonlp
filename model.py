from transformers import T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
import torch

def train_model(train_loader, batch_size=16, num_epochs=3):
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model and move it to the appropriate device
    model = T5ForConditionalGeneration.from_pretrained('t5-small').to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Learning rate scheduler
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            # Move batch data to the appropriate device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            output_ids = batch['output_ids'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=output_ids)
            loss = outputs.loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update the learning rate

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss}")

    # Save the trained model
    model.save_pretrained('./t5_model')

    return model
