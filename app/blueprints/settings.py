from flask import Blueprint, render_template, request, redirect, url_for, session, flash
from app.utils.logger import get_logger
from app.services.user_service import user_service
from app.utils.auth import login_required
from app.blueprints.utils import handle_api_error, success_response, require_json_body
from app.utils.models_config import EMBEDDING_MODELS, LLM_MODELS
from app.services.settings_service import settings_service

logger = get_logger(__name__)

bp = Blueprint("settings", __name__)


@bp.route("/settings")
@login_required
def settings():
    return render_template("settings.html")


@bp.route("/api/v1/settings/models", methods=["GET"])
@handle_api_error
def api_get_available_models():
    return success_response(
        {"embedding_models": EMBEDDING_MODELS, "llm_models": LLM_MODELS}
    )


@bp.route("/api/v1/settings", methods=["GET"])
@handle_api_error
def api_get_settings():
    settings = settings_service.get()
    return success_response(settings)


@bp.route("/api/v1/settings", methods=["PUT"])
@handle_api_error
def api_update_settings():
    data, err = require_json_body()
    if err:
        return err
    settings = settings_service.update(data)
    return success_response(settings, "更新设置成功")
