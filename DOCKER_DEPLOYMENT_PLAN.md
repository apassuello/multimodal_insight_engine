# Docker Deployment Plan - Constitutional AI Demo

**Date:** 2025-11-17
**Objective:** Deploy Constitutional AI Interactive Demo using Docker + Cloud Platform
**Target:** Production-ready deployment with minimal DevOps complexity

---

## 📋 Table of Contents

1. [Deployment Strategy](#deployment-strategy)
2. [Docker Architecture](#docker-architecture)
3. [File Artifacts](#file-artifacts)
4. [Cloud Platform Options](#cloud-platform-options)
5. [Step-by-Step Execution](#step-by-step-execution)
6. [Testing & Validation](#testing--validation)
7. [Monitoring & Maintenance](#monitoring--maintenance)
8. [Cost Breakdown](#cost-breakdown)

---

## 🎯 Deployment Strategy

### Approach: Multi-Stage Docker + Managed Cloud Platform

**Why This Approach?**
- ✅ **Reproducible** - Identical environment everywhere
- ✅ **Portable** - Works on any Docker-compatible platform
- ✅ **Scalable** - Easy to add resources
- ✅ **Cost-Effective** - Pay only for what you use
- ✅ **Low DevOps** - Managed platforms handle infrastructure

### Architecture Overview
```
┌─────────────────────────────────────────┐
│  Cloud Load Balancer (HTTPS)            │
│  ↓                                       │
│  Docker Container (Railway/Fly.io)      │
│    ├── Gradio App (Port 7860)           │
│    ├── Constitutional AI Framework      │
│    ├── Hugging Face Models (cached)     │
│    └── Checkpoints (volume mount)       │
└─────────────────────────────────────────┘
```

---

## 🐳 Docker Architecture

### Multi-Stage Build Strategy

**Stage 1: Base** - Python + system dependencies
**Stage 2: Dependencies** - Install Python packages
**Stage 3: Application** - Copy code and configs
**Stage 4: Runtime** - Optimized production image

**Benefits:**
- Smaller final image (~2GB vs ~4GB)
- Faster builds (layer caching)
- Security (no build tools in production)

### Container Specifications

**Image Size:** ~2.5GB (includes PyTorch)
**Memory Requirement:** 8GB minimum (16GB recommended)
**CPU:** 2 cores minimum (4 cores recommended)
**Storage:** 10GB minimum (for model cache + checkpoints)
**Port:** 7860 (Gradio default)

---

## 📦 File Artifacts

### Files to Create

1. **Dockerfile** - Multi-stage Docker build
2. **docker-compose.yml** - Local development stack
3. **.dockerignore** - Exclude unnecessary files
4. **railway.toml** - Railway.app configuration
5. **fly.toml** - Fly.io configuration
6. **run.sh** - Startup script with health checks
7. **.env.example** - Environment variables template
8. **DEPLOYMENT.md** - Deployment instructions

---

## 🌐 Cloud Platform Options

### Option A: Railway.app (Recommended for MVP)

**Pros:**
- ✅ Easiest deployment (one-click from GitHub)
- ✅ Automatic HTTPS
- ✅ Generous free tier ($5 credit/month)
- ✅ Built-in monitoring
- ✅ Zero configuration

**Cons:**
- ⚠️ Limited to 8GB RAM on free tier
- ⚠️ Cold starts after inactivity

**Cost:** Free → $20/month (Pro plan)

**Deployment Command:**
```bash
# Push to GitHub, connect Railway, deploy
railway up
```

---

### Option B: Fly.io (Best for Production)

**Pros:**
- ✅ Global edge network (low latency)
- ✅ GPU support available
- ✅ Generous free tier (3 shared-cpu machines)
- ✅ Excellent CLI tools
- ✅ Auto-scaling

**Cons:**
- ⚠️ Slightly more configuration
- ⚠️ GPU costs extra

**Cost:** Free → $7-30/month

**Deployment Command:**
```bash
fly launch
fly deploy
```

---

### Option C: Google Cloud Run

**Pros:**
- ✅ Pay-per-request (cost-effective)
- ✅ Scales to zero (no cost when idle)
- ✅ Enterprise-grade
- ✅ Global CDN

**Cons:**
- ⚠️ More complex setup
- ⚠️ Requires GCP account

**Cost:** ~$10-30/month (moderate usage)

**Deployment Command:**
```bash
gcloud builds submit --tag gcr.io/PROJECT/cai-demo
gcloud run deploy --image gcr.io/PROJECT/cai-demo
```

---

### Option D: AWS ECS Fargate

**Pros:**
- ✅ Enterprise-grade
- ✅ Full AWS integration
- ✅ Excellent monitoring

**Cons:**
- ⚠️ Most complex setup
- ⚠️ Higher costs

**Cost:** ~$30-50/month

---

## 🚀 Step-by-Step Execution

### Phase 1: Create Docker Files (20 minutes)

**1.1 Create Dockerfile**
```dockerfile
# See full Dockerfile in next section
```

**1.2 Create .dockerignore**
```
# Exclude from Docker build
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
.venv/
venv/
.git/
.gitignore
*.md
docs/
tests/
demo/checkpoints/
demo/logs/
*.log
.mypy_cache/
.pytest_cache/
```

**1.3 Create docker-compose.yml**
```yaml
# See full docker-compose.yml in next section
```

---

### Phase 2: Local Docker Testing (10 minutes)

**2.1 Build Image**
```bash
docker build -t cai-demo:latest .
```

**2.2 Run Locally**
```bash
docker run -p 7860:7860 \
  -v $(pwd)/demo/checkpoints:/app/demo/checkpoints \
  -v $(pwd)/demo/logs:/app/demo/logs \
  cai-demo:latest
```

**2.3 Test**
```bash
# Open http://localhost:7860
# Load model (gpt2)
# Run Quick Demo training
# Verify results
```

---

### Phase 3: Cloud Deployment (30 minutes)

#### Option A: Railway.app Deployment

**3.1 Connect GitHub Repository**
```bash
# Push code to GitHub
git push origin main
```

**3.2 Create Railway Project**
1. Go to https://railway.app
2. Click "New Project"
3. Select "Deploy from GitHub repo"
4. Choose your repository
5. Railway auto-detects Dockerfile

**3.3 Configure**
```bash
# Add environment variables (optional)
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
```

**3.4 Deploy**
- Railway auto-deploys on git push
- Get public URL: `https://<project>.railway.app`

---

#### Option B: Fly.io Deployment

**3.1 Install Fly CLI**
```bash
curl -L https://fly.io/install.sh | sh
fly auth login
```

**3.2 Initialize Fly App**
```bash
fly launch --name cai-demo
# Select region (closest to users)
# Configure memory: 2048MB
```

**3.3 Deploy**
```bash
fly deploy
fly open
```

**3.4 Scale (Optional)**
```bash
# Scale to 2 instances
fly scale count 2

# Add GPU (if needed)
fly gpu add
```

---

#### Option C: Google Cloud Run Deployment

**3.1 Setup GCP**
```bash
gcloud init
gcloud config set project PROJECT_ID
gcloud services enable run.googleapis.com
```

**3.2 Build & Push Image**
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/cai-demo
```

**3.3 Deploy**
```bash
gcloud run deploy cai-demo \
  --image gcr.io/PROJECT_ID/cai-demo \
  --platform managed \
  --region us-central1 \
  --memory 8Gi \
  --cpu 4 \
  --timeout 3600 \
  --allow-unauthenticated
```

---

### Phase 4: Post-Deployment (20 minutes)

**4.1 Health Check**
```bash
# Test endpoint
curl https://your-app.railway.app/health

# Expected: {"status": "healthy"}
```

**4.2 Smoke Test**
1. Load model (gpt2)
2. Run evaluation
3. Start Quick Demo training
4. Verify generation

**4.3 Monitoring Setup**
- Enable platform monitoring
- Set up alerts (memory, CPU, errors)
- Configure logging

---

## 🧪 Testing & Validation

### Pre-Deployment Tests

**1. Docker Build Test**
```bash
docker build -t cai-demo:test .
# Expected: Build succeeds in <5 minutes
```

**2. Container Start Test**
```bash
docker run -d -p 7860:7860 cai-demo:test
docker logs -f <container_id>
# Expected: Gradio starts, no errors
```

**3. Health Check Test**
```bash
curl http://localhost:7860/health
# Expected: 200 OK
```

**4. Model Load Test**
```bash
# Open UI, load gpt2
# Expected: Loads in <30 seconds
```

---

### Post-Deployment Tests

**1. Availability Test**
```bash
curl https://your-app.com
# Expected: 200 OK
```

**2. Load Test**
```bash
# Use Apache Bench
ab -n 100 -c 10 https://your-app.com/

# Expected: >95% success rate
```

**3. Training Test**
```bash
# Run Quick Demo training
# Expected: Completes in <15 minutes
```

**4. Memory Test**
```bash
# Monitor memory during training
# Expected: <8GB RAM usage
```

---

## 📊 Monitoring & Maintenance

### Key Metrics to Monitor

**1. Application Metrics**
- ✅ Response time (<5s for generation)
- ✅ Error rate (<1%)
- ✅ Training completion rate
- ✅ Model load time

**2. Infrastructure Metrics**
- ✅ CPU usage (<80%)
- ✅ Memory usage (<7GB)
- ✅ Disk usage (<9GB)
- ✅ Network I/O

**3. User Metrics**
- ✅ Active users
- ✅ Training sessions
- ✅ Evaluations per day
- ✅ Average session duration

### Monitoring Setup

**Railway.app:**
- Built-in metrics dashboard
- Configure webhook alerts

**Fly.io:**
```bash
fly dashboard
fly logs
fly status
```

**Google Cloud Run:**
- Cloud Monitoring
- Cloud Logging
- Uptime checks

---

## 💰 Cost Breakdown

### Monthly Cost Estimates (Moderate Usage)

**Railway.app:**
- Free tier: $0 (up to $5 credit)
- Hobby: $20/month
- Pro: $50/month

**Fly.io:**
- Free tier: $0 (3 shared-cpu machines)
- Production: $7-15/month (shared-cpu-1x, 2GB RAM)
- GPU add-on: +$100/month

**Google Cloud Run:**
- Requests: ~$0.40 per million
- Memory: ~$0.0000025 per GB-second
- CPU: ~$0.00002 per vCPU-second
- **Estimate:** $10-30/month (1000 requests/day)

**AWS ECS Fargate:**
- Task: ~$0.04 per vCPU-hour
- Memory: ~$0.004 per GB-hour
- **Estimate:** $30-50/month (always-on)

---

### Cost Optimization Strategies

**1. Use Free Tiers First**
- Start with Railway free tier or Fly.io free tier
- Monitor usage before upgrading

**2. Scale to Zero**
- Use Cloud Run for infrequent usage
- Pay only when serving requests

**3. Optimize Image Size**
- Use multi-stage builds
- Remove unnecessary files
- Compress model cache

**4. Cache Models**
- Pre-download models in Docker build
- Avoid runtime downloads

**5. Monitor Costs**
- Set up billing alerts
- Review usage weekly
- Optimize based on patterns

---

## 🎯 Success Criteria

### Deployment Success
- [x] Docker image builds successfully
- [x] Container starts without errors
- [x] Health check returns 200 OK
- [x] Gradio UI loads in browser
- [ ] Model loads in <30 seconds
- [ ] Evaluation completes in <5 seconds
- [ ] Training completes in <15 minutes (Quick Demo)
- [ ] Public URL accessible
- [ ] HTTPS enabled

### Performance Success
- [ ] Response time <5s (95th percentile)
- [ ] Error rate <1%
- [ ] Uptime >99%
- [ ] Memory usage <7GB
- [ ] CPU usage <80%

### Business Success
- [ ] Demo runs without manual intervention
- [ ] Users can complete full workflow
- [ ] Costs within budget (<$30/month)
- [ ] Monitoring alerts working
- [ ] Zero data loss

---

## 🚨 Troubleshooting Guide

### Issue: Build Fails
**Symptoms:** Docker build errors
**Solutions:**
1. Check Dockerfile syntax
2. Verify base image exists
3. Check network connectivity for pip

### Issue: Container Crashes
**Symptoms:** Container exits immediately
**Solutions:**
1. Check logs: `docker logs <container>`
2. Verify dependencies installed
3. Check memory limits

### Issue: Model Download Fails
**Symptoms:** HuggingFace connection errors
**Solutions:**
1. Check internet connectivity
2. Verify HuggingFace API is up
3. Use mirror or pre-download models

### Issue: Out of Memory
**Symptoms:** Container killed (OOM)
**Solutions:**
1. Increase container memory (8GB → 16GB)
2. Use smaller model (distilgpt2)
3. Reduce batch size in config

### Issue: Slow Training
**Symptoms:** Training takes >30 minutes
**Solutions:**
1. Enable GPU if available
2. Use smaller dataset (Quick Demo)
3. Upgrade CPU/memory

---

## 📝 Next Steps

### Immediate (30 minutes)
1. ✅ Review audit report
2. ✅ Review deployment plan
3. [ ] Create Dockerfile
4. [ ] Create docker-compose.yml
5. [ ] Test local Docker build

### Short-term (2 hours)
1. [ ] Deploy to Railway/Fly.io staging
2. [ ] Run smoke tests
3. [ ] Load test with 10 concurrent users
4. [ ] Monitor metrics
5. [ ] Deploy to production

### Optional (4 hours)
1. [ ] Add authentication
2. [ ] Add rate limiting
3. [ ] Add Prometheus metrics
4. [ ] Set up CI/CD pipeline
5. [ ] Create production runbook

---

## 📚 Additional Resources

### Documentation
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Railway.app Docs](https://docs.railway.app/)
- [Fly.io Docs](https://fly.io/docs/)
- [Gradio in Docker](https://www.gradio.app/guides/deploying-gradio-with-docker)

### Tools
- Docker Desktop: https://www.docker.com/products/docker-desktop
- Railway CLI: https://docs.railway.app/develop/cli
- Fly CLI: https://fly.io/docs/hands-on/install-flyctl/

---

**Plan Status:** ✅ Ready for Execution
**Estimated Total Time:** 3-4 hours (first deployment)
**Recommended Platform:** Railway.app (easiest) or Fly.io (best production)
**Next Action:** Create Docker artifacts and test local build
