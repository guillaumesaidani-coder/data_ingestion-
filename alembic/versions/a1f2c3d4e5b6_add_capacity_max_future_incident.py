"""add_capacity_max_future_incident

Revision ID: a1f2c3d4e5b6
Revises: dc78ff53a32e
Create Date: 2026-09-19 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1f2c3d4e5b6'
down_revision: Union[str, Sequence[str], None] = 'dc78ff53a32e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Rolling max manquants (voltage/rotation) -- symetrie avec temp/pressure
    op.add_column('gold_machine_hourly_feature', sa.Column('voltage_max_6h', sa.Float(), nullable=True))
    op.add_column('gold_machine_hourly_feature', sa.Column('rotation_max_6h', sa.Float(), nullable=True))
    op.add_column('gold_machine_hourly_feature', sa.Column('voltage_max_12h', sa.Float(), nullable=True))
    op.add_column('gold_machine_hourly_feature', sa.Column('rotation_max_12h', sa.Float(), nullable=True))
    op.add_column('gold_machine_hourly_feature', sa.Column('voltage_max_24h', sa.Float(), nullable=True))
    op.add_column('gold_machine_hourly_feature', sa.Column('rotation_max_24h', sa.Float(), nullable=True))

    # Production
    op.add_column('gold_machine_hourly_feature', sa.Column('capacity_utilization_pct', sa.Float(), nullable=True))

    # Compte brut d'incidents futurs (avant seuillage booleen)
    op.add_column('gold_machine_hourly_feature', sa.Column('future_incident_count_6h', sa.Integer(), nullable=True))
    op.add_column('gold_machine_hourly_feature', sa.Column('future_incident_count_12h', sa.Integer(), nullable=True))
    op.add_column('gold_machine_hourly_feature', sa.Column('future_incident_count_24h', sa.Integer(), nullable=True))
    op.add_column('gold_machine_hourly_feature', sa.Column('future_incident_count_48h', sa.Integer(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('gold_machine_hourly_feature', 'future_incident_count_48h')
    op.drop_column('gold_machine_hourly_feature', 'future_incident_count_24h')
    op.drop_column('gold_machine_hourly_feature', 'future_incident_count_12h')
    op.drop_column('gold_machine_hourly_feature', 'future_incident_count_6h')
    op.drop_column('gold_machine_hourly_feature', 'capacity_utilization_pct')
    op.drop_column('gold_machine_hourly_feature', 'rotation_max_24h')
    op.drop_column('gold_machine_hourly_feature', 'voltage_max_24h')
    op.drop_column('gold_machine_hourly_feature', 'rotation_max_12h')
    op.drop_column('gold_machine_hourly_feature', 'voltage_max_12h')
    op.drop_column('gold_machine_hourly_feature', 'rotation_max_6h')
    op.drop_column('gold_machine_hourly_feature', 'voltage_max_6h')
