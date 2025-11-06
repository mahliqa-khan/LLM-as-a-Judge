# Deployment Guide

This guide shows you how to deploy the LLM Judge Evaluator web app to Streamlit Cloud (free hosting).

## Quick Deploy to Streamlit Cloud (Recommended)

### Prerequisites
- GitHub account
- Your code pushed to a GitHub repository

### Step 1: Push to GitHub

```bash
# Initialize git (if not already done)
git init
git add .
git commit -m "Add LLM Judge Evaluator web app"

# Create a new repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### Step 2: Deploy to Streamlit Cloud

1. **Go to** [share.streamlit.io](https://share.streamlit.io)

2. **Sign in** with your GitHub account

3. **Click "New app"**

4. **Fill in the details:**
   - **Repository:** `YOUR_USERNAME/YOUR_REPO`
   - **Branch:** `main`
   - **Main file path:** `app.py`

5. **Click "Deploy"**

6. **Wait 2-3 minutes** for deployment to complete

7. **Done!** Your app will be live at `https://YOUR_USERNAME-YOUR_REPO.streamlit.app`

### Step 3: Share Your App

Share the URL with anyone who needs to evaluate LLM judges. No authentication required!

## Local Development

### Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Test with Sample Data

1. Run the app
2. Click "🎮 Try Sample Data" button
3. Explore the interface with 15 sample evaluations

## Alternative Deployment Options

### Option 2: Hugging Face Spaces (Free)

1. Create account at [huggingface.co](https://huggingface.co)
2. Create new Space (select "Streamlit" as SDK)
3. Push your code to the Space repository
4. App auto-deploys

### Option 3: Railway (Free tier available)

1. Go to [railway.app](https://railway.app)
2. Sign in with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Railway auto-detects Streamlit and deploys

### Option 4: Render (Free tier)

1. Go to [render.com](https://render.com)
2. Sign up and connect GitHub
3. Create new "Web Service"
4. Select your repository
5. Set build command: `pip install -r requirements.txt`
6. Set start command: `streamlit run app.py --server.port $PORT`

## Configuration

### Custom Theme

Edit `.streamlit/config.toml` to customize colors:

```toml
[theme]
primaryColor = "#3498db"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

### Port Configuration

By default, Streamlit runs on port 8501. Change in `.streamlit/config.toml`:

```toml
[server]
port = 8080
```

## Features

Your deployed app includes:

✅ **File Upload** - Upload JSON evaluation data
✅ **Interactive Charts** - Scatter plots, histograms, pie charts
✅ **Statistical Metrics** - Accuracy, correlation, MAE, RMSE
✅ **Error Analysis** - Confusion matrix, bias analysis
✅ **Prediction Review** - View worst/best predictions
✅ **Download Reports** - Export as JSON or text
✅ **Sample Data** - Try the app without uploading files

## Troubleshooting

### "Module not found" error
```bash
# Make sure all dependencies are in requirements.txt
pip install -r requirements.txt
```

### App crashes on Streamlit Cloud
- Check the logs in Streamlit Cloud dashboard
- Ensure `sample_evaluation_data.json` is in the repository
- Verify all imports are in `requirements.txt`

### File upload not working
- Check file format is valid JSON
- Ensure required fields exist: id, model_output, human_score, gpt5_score
- Try the template file first

## Cost

**Streamlit Cloud:**
- ✅ Free for public apps
- ✅ Unlimited usage
- ✅ No credit card required

**Hugging Face Spaces:**
- ✅ Free tier available
- ✅ Good for public demos

**Railway/Render:**
- ⚠️ Free tier with limits
- 💰 Paid plans for more usage

## Security Note

This app runs entirely in the browser and on the server. Uploaded data:
- ❌ Is NOT stored permanently
- ✅ Is processed in memory only
- ✅ Is deleted after analysis

For sensitive data, run locally instead of deploying publicly.

## Support

Need help? Check:
- [Streamlit Documentation](https://docs.streamlit.io)
- [Streamlit Community Forum](https://discuss.streamlit.io)
- Your app's README.md

## Next Steps

Once deployed:
1. Share the URL with your team
2. Upload your 90-100 evaluation samples
3. Review the statistical analysis
4. Download reports for your records

Enjoy your deployed LLM Judge Evaluator! 🎉
