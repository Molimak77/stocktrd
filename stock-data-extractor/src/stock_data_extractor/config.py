# -*- coding: utf-8 -*-
"""
This module handles the configuration for the stock_data_extractor package.
"""

import logging
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    """
    Loads configuration from environment variables stored in a .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )
    yf_api_key: Optional[str] = None


try:
    config = AppConfig()
except Exception as e:
    logging.error(f"Failed to load configuration. Ensure a .env file exists: {e}")
    config = None