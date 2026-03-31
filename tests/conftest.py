import pytest
import pytest_asyncio

from app.services.db_service import init_db


@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_db():
    await init_db()