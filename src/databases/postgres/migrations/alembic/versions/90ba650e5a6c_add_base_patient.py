"""add base patient

Revision ID: 90ba650e5a6c
Revises: bdd7bbb8443d
Create Date: 2025-12-11 22:02:05.171652

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision = '90ba650e5a6c'
down_revision = 'bdd7bbb8443d'
branch_labels = None
depends_on = None

_SEED_NAME = f"base_patient__seed__{revision}"


def upgrade():
    conn = op.get_bind()

    exists = conn.execute(text("SELECT EXISTS (SELECT 1 FROM patient LIMIT 1)")).scalar()

    if not exists:
        conn.execute(text(f"INSERT INTO patient (name) VALUES ('{_SEED_NAME}')"))


def downgrade():
    conn = op.get_bind()
    conn.execute(text(f"DELETE FROM patient WHERE name = '{_SEED_NAME}'"))
