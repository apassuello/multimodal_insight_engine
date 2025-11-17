# Deployment Guide - Constitutional AI Demo

**Quick Start:** Get the demo running in production in under 1 hour

---

## 🚀 Quick Deploy Options

### Option 1: Railway.app (Easiest - 10 minutes)

**Prerequisites:** GitHub account, Railway account

```bash
# 1. Push code to GitHub
git add .
git commit -m "Add Docker deployment"
git push origin main

# 2. Go to https://railway.app
# 3. Click "New Project" → "Deploy from GitHub repo"
# 4. Select your repository
# 5. Railway auto-detects Dockerfile and deploys!

# Your app will be live at: https://<project>.railway.app
```

**Cost:** Free tier ($5 credit/month) → $20/month Pro

---

### Option 2: Fly.io (Best for Production - 15 minutes)

**Prerequisites:** Fly.io account, Fly CLI installed

```bash
# 1. Install Fly CLI
curl -L https://fly.io/install.sh | sh

# 2. Login
fly auth login

# 3. Launch app (creates fly.toml automatically)
fly launch --name cai-demo

# Select region closest to your users
# Configure: 2048 MB RAM, 2 CPU cores

# 4. Deploy
fly deploy

# 5. Open in browser
fly open
```

**Cost:** Free tier (3 shared-cpu machines) → $7-15/month

---

### Option 3: Local Docker (Testing - 5 minutes)

**Prerequisites:** Docker Desktop installed

```bash
# 1. Build the image
docker build -t cai-demo:latest .

# 2. Run with docker-compose
docker-compose up

# 3. Open browser
open http://localhost:7860

# To stop:
docker-compose down
```

---

## 📦 What's Included

All Docker deployment files are ready:

- ✅ `Dockerfile` - Multi-stage production build
- ✅ `docker-compose.yml` - Local development stack
- ✅ `.dockerignore` - Optimized build context
- ✅ `run.sh` - Startup script with health checks
- ✅ `.env.example` - Environment configuration template

---

## 🔧 Configuration

### Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

**Key variables:**
- `GRADIO_SERVER_NAME=0.0.0.0` - Server hostname
- `GRADIO_SERVER_PORT=7860` - Server port
- `DEFAULT_MODEL=gpt2` - Default model to use
- `DEVICE_PREFERENCE=auto` - Device preference (auto/mps/cuda/cpu)

### Volume Mounts

The demo persists data in these directories:
- `demo/checkpoints/` - Model checkpoints (base & trained)
- `demo/logs/` - Training logs
- `demo/exports/` - Export data

**Docker Compose** automatically mounts these as volumes.

---

## 🧪 Testing

### 1. Test Docker Build

```bash
docker build -t cai-demo:test .
```

**Expected:** Build completes in ~5 minutes

### 2. Test Local Run

```bash
docker-compose up
```

**Expected:**
- Gradio starts on http://localhost:7860
- No errors in logs
- Can load model (gpt2)

### 3. Smoke Test

1. Open http://localhost:7860
2. Click "Load Model" → Select "gpt2" → Click "Load"
3. Go to "Evaluation" tab → Load example → Click "Evaluate"
4. Verify results appear

---

## 🌐 Cloud Platform Details

### Railway.app

**Setup:**
1. Connect GitHub repository
2. Railway auto-detects Dockerfile
3. Auto-deploys on git push

**Features:**
- Automatic HTTPS
- Built-in monitoring
- Zero configuration
- $5/month free credit

**Recommended for:** MVP, quick demos, non-critical apps

---

### Fly.io

**Setup:**
```bash
fly launch --name cai-demo
fly deploy
```

**Features:**
- Global edge network (low latency)
- GPU support available
- Excellent CLI tools
- Auto-scaling

**Recommended for:** Production, global apps, GPU workloads

**Scaling:**
```bash
# Scale to 2 instances
fly scale count 2

# Increase memory
fly scale memory 4096

# Add GPU (if needed)
fly gpu add
```

---

### Google Cloud Run

**Setup:**
```bash
# Build and push image
gcloud builds submit --tag gcr.io/PROJECT_ID/cai-demo

# Deploy
gcloud run deploy cai-demo \
  --image gcr.io/PROJECT_ID/cai-demo \
  --platform managed \
  --region us-central1 \
  --memory 8Gi \
  --timeout 3600 \
  --allow-unauthenticated
```

**Features:**
- Pay-per-request (cost-effective)
- Scales to zero
- Enterprise-grade
- Global CDN

**Recommended for:** Variable traffic, enterprise deployments

---

## 📊 Resource Requirements

### Minimum (CPU-only demo)
- **Memory:** 4GB
- **CPU:** 2 cores
- **Storage:** 5GB

### Recommended (Production)
- **Memory:** 8GB
- **CPU:** 4 cores
- **Storage:** 10GB

### Optimal (GPU-accelerated)
- **Memory:** 16GB
- **CPU:** 8 cores
- **GPU:** T4 or better
- **Storage:** 20GB

---

## 🔒 Security Checklist

For public deployments:

- [ ] Enable HTTPS (automatic on Railway/Fly.io)
- [ ] Add authentication (Gradio auth or external)
- [ ] Configure rate limiting
- [ ] Set resource limits (memory/CPU)
- [ ] Enable request logging
- [ ] Monitor for abuse patterns
- [ ] Validate input lengths
- [ ] Use secure environment variables

---

## 🐛 Troubleshooting

### Build fails with "No space left on device"
**Solution:** Clean Docker cache
```bash
docker system prune -a
```

### Container exits immediately
**Solution:** Check logs
```bash
docker logs <container-id>
```

### Model download fails
**Solution:** Check internet connectivity or pre-download models
```bash
# Pre-download model
python -c "from transformers import AutoModel; AutoModel.from_pretrained('gpt2')"
```

### Out of memory (OOM)
**Solutions:**
1. Increase container memory to 8GB+
2. Use smaller model (distilgpt2)
3. Reduce batch size in config.yaml

### Slow training
**Solutions:**
1. Enable GPU if available
2. Use Quick Demo mode (2 epochs, 20 examples)
3. Upgrade CPU/memory

---

## 📈 Monitoring

### Health Check
```bash
curl https://your-app.railway.app/health
```

### View Logs

**Railway:**
```
# View in dashboard
https://railway.app/project/<id>/deployments
```

**Fly.io:**
```bash
fly logs
```

**Docker:**
```bash
docker logs -f <container-id>
```

### Metrics to Monitor
- Response time (target: <5s)
- Memory usage (target: <7GB)
- Error rate (target: <1%)
- Training completion rate

---

## 💰 Cost Estimates

### Free Tier Options
- **Railway:** $5 credit/month (limited hours)
- **Fly.io:** 3 shared-cpu machines (always free)
- **Google Cloud:** $300 credit (90 days)

### Paid Tiers (Monthly)
- **Railway Pro:** $20 + usage
- **Fly.io Production:** $7-15 (shared-cpu-1x, 2GB)
- **Fly.io GPU:** +$100 (T4)
- **Cloud Run:** $10-30 (moderate usage)

**Recommendation:** Start with free tier, upgrade based on usage

---

## 🎯 Success Checklist

- [ ] Docker image builds successfully
- [ ] Container starts without errors
- [ ] Gradio UI loads in browser
- [ ] Model loads in <30 seconds
- [ ] Evaluation works (AI + regex)
- [ ] Training completes (Quick Demo)
- [ ] Generation compares base vs trained
- [ ] Public URL accessible (cloud deployment)
- [ ] HTTPS enabled
- [ ] Monitoring configured

---

## 📚 Additional Resources

- [Demo Audit Report](DEMO_AUDIT_REPORT.md) - Complete demo analysis
- [Docker Deployment Plan](DOCKER_DEPLOYMENT_PLAN.md) - Detailed deployment strategy
- [Demo Architecture](DEMO_ARCHITECTURE.md) - Technical architecture
- [User Guide](docs/USER_GUIDE.md) - How to use the demo

---

## 🆘 Support

**Issues?**
1. Check troubleshooting section above
2. Review logs for error messages
3. Consult deployment platform docs
4. Open GitHub issue with logs

---

**Last Updated:** 2025-11-17
**Deployment Status:** ✅ Production Ready
**Estimated Setup Time:** 10-60 minutes (depending on platform)
