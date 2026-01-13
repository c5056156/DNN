# train.py
import torch
from transformers import BertModel, BertTokenizer, CLIPModel, CLIPProcessor
from torch.optim import Adam
from tqdm import tqdm

from model import Generator, Discriminator
from utils import adversarial_loss, recon_loss, set_seed

# -------------------------
# Config
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50
LR = 2e-4

set_seed(42)

# -------------------------
# Load text + CLIP
# -------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
text_encoder = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)
text_encoder.eval()

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

# -------------------------
# Models
# -------------------------
G = Generator(text_encoder).to(DEVICE)
D = Discriminator().to(DEVICE)

g_opt = Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
d_opt = Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

# -------------------------
# Dataset loader (YOU PROVIDE)
# -------------------------
# train_loader = ...

# -------------------------
# Training loop
# -------------------------
g_losses, d_losses, clip_scores = [], [], []
global_step = 0

for epoch in range(EPOCHS):
    G.train(); D.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for batch in pbar:
        imgs = batch["imgs"].to(DEVICE)
        input_ids = batch["input_ids"].to(DEVICE)
        att_mask = batch["attention_mask"].to(DEVICE)
        targets = batch["targets"].to(DEVICE)
        B = imgs.size(0)

        # ----- Discriminator -----
        d_opt.zero_grad()
        with torch.no_grad():
            fake_imgs = G(imgs, input_ids, att_mask)

        real_preds = D(targets)
        fake_preds = D(fake_imgs.detach())

        real_labels = torch.ones_like(real_preds)
        fake_labels = torch.zeros_like(fake_preds)

        d_loss = 0.5 * (
            adversarial_loss(real_preds, real_labels) +
            adversarial_loss(fake_preds, fake_labels)
        )
        d_loss.backward()
        d_opt.step()

        # ----- Generator -----
        g_opt.zero_grad()
        gen_imgs = G(imgs, input_ids, att_mask)
        pred_logits = D(gen_imgs)

        g_adv = adversarial_loss(pred_logits, real_labels)
        g_l1 = recon_loss(gen_imgs, targets)
        g_loss = g_adv + 10.0 * g_l1

        g_loss.backward()
        g_opt.step()

        pbar.set_postfix({
            "g_loss": g_loss.item(),
            "d_loss": d_loss.item()
        })

        global_step += 1

    torch.save({
        "G": G.state_dict(),
        "D": D.state_dict()
    }, f"checkpoints_seq/epoch_{epoch}.pt")

print("Training complete.")
