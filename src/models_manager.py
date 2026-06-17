"""
Models Manager - 从 CodeBuddy 查询真实可用模型并持久化结果

通过调用 https://copilot.tencent.com/v3/config 获取 CodeBuddy 真实支持的模型列表。
返回数据中 data.models 包含每个模型的 id、name、description 等完整信息。

可用模型列表会保存到 config/available_models.json，
前端 /v1/models 优先返回此列表。
"""
import os
import json
import time
import logging
from typing import Dict, Any, List, Optional

import httpx

logger = logging.getLogger(__name__)

_AVAILABLE_MODELS_FILE = 'config/available_models.json'
# CodeBuddy 配置接口（与聊天接口不同域名）
_CONFIG_API_URL = 'https://copilot.tencent.com/v3/config'


class ModelsManager:
    """模型查询与持久化管理器"""

    def __init__(self):
        self._cache: Optional[Dict[str, Any]] = None
        _refreshing = False

    # ---------- 持久化 ----------

    def _load_saved(self) -> Optional[Dict[str, Any]]:
        """从磁盘加载已保存的可用模型列表"""
        if self._cache is not None:
            return self._cache
        if os.path.exists(_AVAILABLE_MODELS_FILE):
            try:
                with open(_AVAILABLE_MODELS_FILE, 'r', encoding='utf-8') as f:
                    self._cache = json.load(f)
                    return self._cache
            except Exception as e:
                logger.warning(f"读取已保存的可用模型列表失败: {e}")
        return None

    def _save(self, data: Dict[str, Any]) -> None:
        """保存可用模型列表到磁盘"""
        try:
            config_dir = os.path.dirname(_AVAILABLE_MODELS_FILE)
            if not os.path.exists(config_dir):
                os.makedirs(config_dir)
            with open(_AVAILABLE_MODELS_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self._cache = data
            logger.info(f"可用模型列表已保存到 {_AVAILABLE_MODELS_FILE}")
        except Exception as e:
            logger.error(f"保存可用模型列表失败: {e}")

    def clear_saved(self) -> None:
        """清除已保存的列表（回退到配置的完整列表）"""
        self._cache = None
        if os.path.exists(_AVAILABLE_MODELS_FILE):
            try:
                os.remove(_AVAILABLE_MODELS_FILE)
                logger.info("已清除保存的可用模型列表")
            except Exception as e:
                logger.warning(f"清除可用模型列表文件失败: {e}")

    # ---------- 查询 ----------

    def get_available_models(self) -> List[str]:
        """获取可用模型 ID 列表。优先返回查询结果，否则回退到配置的完整列表。"""
        saved = self._load_saved()
        if saved and saved.get('models'):
            return [m['id'] for m in saved['models']]
        # 回退：从 config 读取配置的完整模型列表
        from config import get_available_models
        return get_available_models()

    def get_available_models_detail(self) -> Optional[Dict[str, Any]]:
        """获取可用模型的详细信息（含查询时间、每个模型信息）"""
        return self._load_saved()

    # ---------- 查询 CodeBuddy ----------

    @property
    def is_refreshing(self) -> bool:
        return getattr(self, '_refreshing', False)

    async def fetch_models_from_codebuddy(self) -> Dict[str, Any]:
        """
        调用 CodeBuddy /v3/config 接口获取真实支持的模型列表。

        Returns:
            {
                "models": [{"id": "...", "name": "...", "description": "...", ...}],
                "probed_at": <timestamp>,
                "total": <count>
            }
        """
        if getattr(self, '_refreshing', False):
            raise RuntimeError("已有查询任务在进行中")

        self._refreshing = True
        try:
            from .codebuddy_api_client import codebuddy_api_client
            from .codebuddy_token_manager import codebuddy_token_manager

            # 获取一个有效凭证
            credential = codebuddy_token_manager.get_next_credential()
            if not credential:
                raise RuntimeError("没有可用的 CodeBuddy 凭证，无法查询模型")

            bearer_token = credential.get('bearer_token')
            user_id = credential.get('user_id')

            # 构造请求头 - /v3/config 需要 IDE 类型的头
            headers = codebuddy_api_client.generate_codebuddy_headers(
                bearer_token=bearer_token,
                user_id=user_id
            )
            headers['X-IDE-Type'] = 'CodeBuddyIDE'
            headers['X-IDE-Name'] = 'CodeBuddyIDE'
            headers['X-IDE-Version'] = '4.9.13'
            headers['X-Product-Version'] = '4.9.13'
            headers['Accept'] = 'application/json'
            headers['Host'] = 'copilot.tencent.com'

            timeout = httpx.Timeout(30.0, connect=10.0, read=30.0)
            async with httpx.AsyncClient(timeout=timeout, verify=False, trust_env=False) as client:
                resp = await client.get(_CONFIG_API_URL, headers=headers)

            if resp.status_code != 200:
                raise RuntimeError(f"CodeBuddy 配置接口返回 HTTP {resp.status_code}: {resp.text[:200]}")

            body = resp.json()
            if body.get('code') != 0:
                raise RuntimeError(f"CodeBuddy 配置接口返回错误: code={body.get('code')}, msg={body.get('msg')}")

            raw_models = body.get('data', {}).get('models', [])
            if not raw_models:
                raise RuntimeError("CodeBuddy 配置接口返回的模型列表为空")

            # 只保留需要的字段，避免保存过大的数据（如 iconUrl base64）
            models = []
            for m in raw_models:
                model_info = {
                    'id': m.get('id', ''),
                    'name': m.get('name', m.get('id', '')),
                    'description': m.get('descriptionZh') or m.get('descriptionEn') or '',
                    'max_input_tokens': m.get('maxInputTokens'),
                    'max_output_tokens': m.get('maxOutputTokens'),
                    'supports_images': m.get('supportsImages', False),
                    'supports_tool_call': m.get('supportsToolCall', False),
                    'vendor': m.get('vendor', ''),
                }
                models.append(model_info)

            data = {
                'models': models,
                'probed_at': int(time.time()),
                'total': len(models),
                'source': 'copilot.tencent.com/v3/config'
            }
            self._save(data)
            return data
        finally:
            self._refreshing = False


# 全局实例
models_manager = ModelsManager()
