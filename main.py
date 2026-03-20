import uvicorn
from core.config import load_config

if __name__ == "__main__":
    cfg = load_config()
    uvicorn.run(
        "api.server:app",
        host=cfg.server.host,
        port=cfg.server.port,
        reload=False,
        log_level="info",
    )
