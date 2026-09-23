"""add_maintenance_tables

Revision ID: dc78ff53a32e
Revises: 4cfc3216b636
Create Date: 2026-09-16 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'dc78ff53a32e'
down_revision: Union[str, Sequence[str], None] = '4cfc3216b636'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('bronze_maintenance',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('ingestion_batch_id', sa.UUID(), nullable=True),
    sa.Column('maintenance_id', sa.Integer(), nullable=True),
    sa.Column('machine_id', sa.String(), nullable=True),
    sa.Column('maintenance_at', sa.String(), nullable=True),
    sa.Column('maintenance_type', sa.String(), nullable=True),
    sa.Column('action_type', sa.String(), nullable=True),
    sa.Column('component', sa.String(), nullable=True),
    sa.Column('description', sa.String(), nullable=True),
    sa.Column('related_incident_id', sa.String(), nullable=True),
    sa.Column('duration_hours', sa.Float(), nullable=True),
    sa.Column('parse_ok', sa.Boolean(), server_default='true', nullable=False),
    sa.Column('parse_ok_reason', sa.String(), server_default='', nullable=False),
    sa.ForeignKeyConstraint(['ingestion_batch_id'], ['ingestion_batch.ingestion_batch_id'], ),
    sa.PrimaryKeyConstraint('id')
    )

    op.create_table('silver_maintenance',
    sa.Column('silver_maintenance_id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('ingestion_batch_id', sa.UUID(), nullable=True),
    sa.Column('maintenance_code', sa.String(), nullable=True),
    sa.Column('machine_id', sa.String(), nullable=True),
    sa.Column('performed_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('maintenance_type', sa.String(), nullable=True),
    sa.Column('action_type', sa.String(), nullable=True),
    sa.Column('component', sa.String(), nullable=True),
    sa.Column('duration_hours', sa.Float(), nullable=True),
    sa.Column('related_incident_code', sa.String(), nullable=True),
    sa.ForeignKeyConstraint(['ingestion_batch_id'], ['ingestion_batch.ingestion_batch_id'], ),
    sa.PrimaryKeyConstraint('silver_maintenance_id'),
    sa.UniqueConstraint('maintenance_code')
    )

    op.add_column('gold_machine_hourly_feature', sa.Column('days_since_last_maintenance', sa.Float(), nullable=True))
    op.add_column('gold_machine_hourly_feature', sa.Column('maintenance_count_prev_30d', sa.Integer(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('gold_machine_hourly_feature', 'maintenance_count_prev_30d')
    op.drop_column('gold_machine_hourly_feature', 'days_since_last_maintenance')
    op.drop_table('silver_maintenance')
    op.drop_table('bronze_maintenance')
