import argparse
import sys
import traceback
from io import BytesIO
from pathlib import Path

import light_side as ls
import numpy as np
import torch
import uvicorn
from PIL import Image
from starlette.responses import RedirectResponse, StreamingResponse

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

tags_metadata = [
    {
        "name": "Enhance",
        "description": "Enhancer dark images to light images",
        "externalDocs": {
            "description": "External Docs for Library: ",
            "url": "https://light-side.readthedocs.io/",
        },
    },
]

app = FastAPI(
    title="Light Side", swagger_ui_parameters={"defaultModelsExpandDepth": -1}
)


def custom_openapi():
    openapi_schema = get_openapi(
        title="Light Side API",
        version=ls.__version__,
        description=ls.__description__,
        routes=app.routes,
        tags=tags_metadata,
        license_info={
            "name": ls.__license__,
            "url": ls.__license_url__,
        },
        contact={
            "name": ls.__author__,
        },
    )
    openapi_schema["info"]["x-logo"] = {
        "url": "https://raw.githubusercontent.com/canturan10/light_side/master/src/light_side.png"
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_LIST = []
for model in ls.available_models():
    for version in ls.get_model_versions(model):
        MODEL_LIST.append(f"{model}-{version}")


@app.on_event("startup")
def load_artifacts():
    app.state.models = {}
    for model in ls.available_models():
        for version in ls.get_model_versions(model):
            app.state.models[f"{model}-{version}"] = ls.Enhancer.from_pretrained(
                model,
                version,
            )
            app.state.models[f"{model}-{version}"].eval()
            app.state.models[f"{model}-{version}"].to(
                "cuda" if torch.cuda.is_available() else "cpu"
            )


@app.on_event("shutdown")
def empty_cache():
    # clear Cuda memory
    torch.cuda.empty_cache()


def read_imagefile(data) -> Image.Image:
    image = Image.open(BytesIO(data))
    return image


@app.get("/", include_in_schema=False)
def main():
    return RedirectResponse(url="/docs")


@app.post("/enhance/", tags=["Enhance"], summary="Low Light Image Enhancement")
async def enhance(
    file: UploadFile = File(...),
    model: str = Query(MODEL_LIST[0], enum=MODEL_LIST),
):
    if file.content_type.startswith("image/") is False:
        raise HTTPException(
            status_code=400,
            detail=f"File '{file.filename}' is not an image.",
        )

    try:
        content = await file.read()
        image = np.array(read_imagefile(content).convert("RGB"))
        results = app.state.models[model].predict(image)
        enhanced_image = Image.fromarray(results[0]["enhanced"])
        buf = BytesIO()
        enhanced_image.save(buf, format="PNG")
        byte_im = buf.getvalue()

        return StreamingResponse(BytesIO(byte_im), media_type="image/png")
    except Exception:
        sys.stdout.flush()
        e_info = sys.exc_info()[1]
        traceback.print_exc()
        return HTTPException(
            status_code=500,
            detail=str(e_info),
            headers={"Content-Type": "text/plain"},
        )


if __name__ == "__main__":
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Runs the API server.")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to run the API on.",
    )
    parser.add_argument(
        "--port",
        help="The port to listen for requests on.",
        type=int,
        default=8500,
    )
    parser.add_argument(
        "--workers",
        help="Number of workers to use.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--reload",
        help="Reload the model on each request.",
        action="store_true",
    )
    parser.add_argument(
        "--use-colors",
        help="Enable user-friendly color output.",
        action="store_true",
    )

    args = parser.parse_args()
    uvicorn.run(
        f"{Path(__file__).stem}:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        use_colors=args.use_colors,
    )
