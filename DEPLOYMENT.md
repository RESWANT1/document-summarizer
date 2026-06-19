# Deployment Guide

This guide covers multiple deployment options for the Document Summarization System.

## 📦 Option 1: GitHub Pages (Demo/Documentation Only)

GitHub Pages can only host static sites, so you'll need to deploy the backend elsewhere and link to it.

### Steps:
1. Create a `docs/` folder with static demo page
2. Enable GitHub Pages in repository settings
3. Link to your deployed backend API

## 🐳 Option 2: Docker Deployment

### Create Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create uploads directory
RUN mkdir -p uploads

# Expose port
EXPOSE 5000

# Run application
CMD ["python", "app.py"]
```

### Build and Run
```bash
docker build -t document-summarizer .
docker run -p 5000:5000 document-summarizer
```

## ☁️ Option 3: Deploy to Heroku (Free Tier)

### Prerequisites
- Heroku account
- Heroku CLI installed

### Steps:

1. **Create Procfile**
```
web: python app.py
```

2. **Create runtime.txt**
```
python-3.9.18
```

3. **Add Aptfile** (for Tesseract)
```
tesseract-ocr
tesseract-ocr-eng
```

4. **Deploy**
```bash
heroku login
heroku create your-app-name
git push heroku main
heroku open
```

**Note**: Heroku's free tier has limited memory. Large models may not work.

## 🚀 Option 4: Deploy to Render (Recommended)

Render offers better free tier than Heroku for ML apps.

### Steps:

1. **Sign up at [render.com](https://render.com)**

2. **Connect GitHub repository**

3. **Create Web Service** with settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
   - **Environment**: Python 3
   - **Instance Type**: Free (or Starter for better performance)

4. **Add Environment Variables**:
   ```
   FLASK_DEBUG=False
   PORT=5000
   ```

5. **Deploy** - Render auto-deploys on git push

**Pros**: 
- Free tier available
- Better for ML models than Heroku
- Auto SSL certificates
- Easy CI/CD

## ☁️ Option 5: AWS EC2 Deployment

For production with GPU support.

### Steps:

1. **Launch EC2 Instance**
   - AMI: Deep Learning AMI (Ubuntu)
   - Instance: g4dn.xlarge (for GPU) or t3.large (CPU only)
   - Security Group: Allow HTTP (80), HTTPS (443), SSH (22)

2. **SSH into instance**
```bash
ssh -i your-key.pem ubuntu@your-ec2-ip
```

3. **Clone and setup**
```bash
git clone https://github.com/YOUR_USERNAME/document-summarizer.git
cd document-summarizer
pip install -r requirements.txt
```

4. **Run with production server**
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

5. **Setup Nginx reverse proxy** (optional)

## 🔵 Option 6: Azure Web Apps

### Steps:

1. **Install Azure CLI**
```bash
az login
```

2. **Create resource group**
```bash
az group create --name summarizer-rg --location eastus
```

3. **Create App Service plan**
```bash
az appservice plan create --name summarizer-plan --resource-group summarizer-rg --sku B1 --is-linux
```

4. **Create web app**
```bash
az webapp create --resource-group summarizer-rg --plan summarizer-plan --name your-app-name --runtime "PYTHON|3.9"
```

5. **Deploy from GitHub**
```bash
az webapp deployment source config --name your-app-name --resource-group summarizer-rg --repo-url https://github.com/YOUR_USERNAME/document-summarizer --branch main
```

## 🎯 Option 7: Google Cloud Run (Serverless)

Best for serverless deployment with auto-scaling.

### Steps:

1. **Install gcloud CLI**

2. **Build container**
```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT/summarizer
```

3. **Deploy to Cloud Run**
```bash
gcloud run deploy summarizer \
  --image gcr.io/YOUR_PROJECT/summarizer \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --timeout 300
```

## 📊 Option 8: Hugging Face Spaces (For Demo)

Perfect for showcasing ML projects to recruiters!

### Steps:

1. **Create new Space** at [huggingface.co/spaces](https://huggingface.co/spaces)

2. **Choose Gradio or Streamlit** SDK

3. **Create app.py** for Gradio:
```python
import gradio as gr
from summarizer.summarizer_service import SummarizerService

summarizer = SummarizerService()

def summarize(file, length):
    # Your summarization logic
    summary, _ = summarizer.summarize(text, length=length)
    return summary

interface = gr.Interface(
    fn=summarize,
    inputs=[
        gr.File(label="Upload Document"),
        gr.Radio(["short", "medium", "long"], label="Length")
    ],
    outputs=gr.Textbox(label="Summary"),
    title="Document Summarizer"
)

interface.launch()
```

4. **Push to Space** (works like GitHub)

**Benefits**:
- Free hosting
- ML-focused platform
- Great for portfolios
- Easy sharing with recruiters

## 🎓 Recommended for Recruiters

For showcasing to recruiters, I recommend:

1. **Primary**: Deploy on **Render** or **Hugging Face Spaces**
   - Free, reliable, and ML-friendly
   - Professional URLs
   - Easy to demonstrate

2. **Backup**: Keep code on **GitHub** with excellent README
   - Shows code quality
   - Demonstrates documentation skills
   - Includes evaluation results

3. **Demo Video**: Create a 2-3 minute demo video
   - Upload to YouTube/Loom
   - Link in README
   - Shows functionality if server is down

## 📝 Pre-Deployment Checklist

- [ ] Update README with your information
- [ ] Remove sensitive data (API keys, personal info)
- [ ] Test all features locally
- [ ] Verify requirements.txt is complete
- [ ] Add screenshots to README
- [ ] Create demo video
- [ ] Test on mobile devices
- [ ] Add error handling for edge cases
- [ ] Setup monitoring/logging
- [ ] Create backup deployment

## 🔍 Monitoring & Maintenance

After deployment:
- Monitor usage and errors
- Keep dependencies updated
- Respond to GitHub issues
- Update README with any changes
- Add GitHub stars/forks to resume

---

Need help with deployment? Create an issue on GitHub!
