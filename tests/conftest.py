import pytest
from app.services.db_service import init_db


@pytest.fixture(scope="session", autouse=True)
async def setup_db():
    await init_db()