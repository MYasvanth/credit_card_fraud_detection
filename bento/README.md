# BentoML Deployment (Optional)

This directory contains BentoML-related files for alternative deployment option.

## Files:

**Configuration:**
- `bentofile.yaml` - Main BentoML configuration
- `production_bentofile.yaml` - Production deployment config
- `simple_bentofile.yaml` - Simple deployment config
- `docker-compose.bento.yml` - Docker compose for BentoML

**Services:**
- `bento_service.py` - BentoML service implementation
- `bento_flask_service.py` - Flask-based BentoML service
- `enhanced_service.py` - Enhanced BentoML service
- `simple_bento.py` - Simple BentoML implementation

**Scripts:**
- `bento_builder.py` - BentoML builder utility
- `deploy_bento.py` - BentoML deployment script
- `run_bento_simple.py` - Script to run simple BentoML service
- `run_enhanced_bento.py` - Script to run enhanced BentoML service

**Batch Files:**
- `run_bento.bat` - Run BentoML service
- `start_bento_8080.bat` - Start BentoML on port 8080
- `start_bento.bat` - Start BentoML service

**Tests:**
- `test_bento_service.py` - BentoML service tests
- `quick_test.py` - Quick BentoML service testing

**Additional Services:**
- `service.py` - Main BentoML service definition

## Note:
**BentoML is NOT required for this project.** 

The main deployment uses:
- **FastAPI** (`src/api/main.py`) for REST API
- **MLflow** for model registry and serving

Use BentoML only if you specifically need its features.