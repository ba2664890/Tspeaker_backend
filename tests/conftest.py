import django
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "core.settings")
django.setup()

import pytest
@pytest.fixture(scope="session")
def django_db_setup():
    pass
