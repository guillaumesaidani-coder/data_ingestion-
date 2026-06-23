from sqlalchemy import Column, String, Float, Integer, SmallInteger, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql import text


class Base(DeclarativeBase):
    pass


class IngestionBatch(Base):
    __tablename__ = "ingestion_batch"

    ingestion_batch_id = Column(UUID(as_uuid=True), primary_key=True,
                                server_default=text("gen_random_uuid()"))
    source_name        = Column(String)
    source_file        = Column(String)
    started_at         = Column(DateTime(timezone=True))
    finished_at        = Column(DateTime(timezone=True))
    rows_read          = Column(Integer)
    rows_loaded        = Column(Integer)
    rows_rejected      = Column(Integer)
    status             = Column(String)


class Operator(Base):
    __tablename__ = "operator"

    operator_id  = Column(Integer, primary_key=True, autoincrement=True)
    operator_key = Column(String, unique=True)
    badge_hash   = Column(String)
    is_active    = Column(Boolean, nullable=False, server_default="true")


class DataQualityIssue(Base):
    __tablename__ = "data_quality_issue"

    dq_issue_id        = Column(Integer, primary_key=True, autoincrement=True)
    ingestion_batch_id = Column(UUID(as_uuid=True),
                                ForeignKey("ingestion_batch.ingestion_batch_id"))
    dataset_name       = Column(String)
    rule_code          = Column(String)
    severity           = Column(String)
    entity_key         = Column(String)
    details            = Column(Text)


class BronzeTelemetry(Base):
    __tablename__ = "bronze_telemetry"

    id                 = Column(Integer, primary_key=True, autoincrement=True)
    ingestion_batch_id = Column(UUID(as_uuid=True),
                                ForeignKey("ingestion_batch.ingestion_batch_id"))
    machine_id         = Column(String)
    timestamp          = Column(String)
    temperature_c      = Column(Float)
    pressure_bar       = Column(Float)
    voltage_mean_v     = Column(Float)
    rotation_mean_rpm  = Column(Float)
    pieces_produced    = Column(Integer)
    parse_ok           = Column(Boolean, nullable=False, server_default="true")
    parse_ok_reason    = Column(String,  nullable=False, server_default="")


class BronzeIncident(Base):
    __tablename__ = "bronze_incidents"

    id                     = Column(Integer, primary_key=True, autoincrement=True)
    ingestion_batch_id     = Column(UUID(as_uuid=True),
                                    ForeignKey("ingestion_batch.ingestion_batch_id"))
    incident_id            = Column(String)
    date                   = Column(String)
    time                   = Column(String)
    operator_name          = Column(String)
    machine_id             = Column(String)
    severity               = Column(Integer)
    operator_badge         = Column(String)
    comment                = Column(String)
    shift                  = Column(String)
    type_surchauffe        = Column(Integer)
    type_baisse_pression   = Column(Integer)
    type_vibration         = Column(Integer)
    type_bruit_mecanique   = Column(Integer)
    type_surconsommation   = Column(Integer)
    type_blocage_mecanique = Column(Integer)
    type_alarme_capteur    = Column(Integer)
    type_arret_urgence     = Column(Integer)
    type_defaut_qualite    = Column(Integer)
    parse_ok               = Column(Boolean, nullable=False, server_default="true")
    parse_ok_reason        = Column(String,  nullable=False, server_default="")


class SilverSensorReading(Base):
    __tablename__ = "silver_sensor_reading"

    sensor_reading_id  = Column(Integer, primary_key=True, autoincrement=True)
    ingestion_batch_id = Column(UUID(as_uuid=True),
                                ForeignKey("ingestion_batch.ingestion_batch_id"))
    machine_id         = Column(String)
    observed_at        = Column(DateTime(timezone=True))
    sensor_type        = Column(String)   # temperature_c | pressure_bar | voltage_mean_v | rotation_mean_rpm | pieces_produced
    sensor_value       = Column(Float)
    unit               = Column(String)
    is_missing         = Column(Boolean, nullable=False, server_default="false")
    is_duplicate       = Column(Boolean, nullable=False, server_default="false")
    is_outlier         = Column(Boolean, nullable=False, server_default="false")


class SilverIncident(Base):
    __tablename__ = "silver_incident"

    silver_incident_id = Column(Integer, primary_key=True, autoincrement=True)
    ingestion_batch_id = Column(UUID(as_uuid=True),
                                ForeignKey("ingestion_batch.ingestion_batch_id"))
    incident_code      = Column(String, unique=True)
    machine_id         = Column(String)
    operator_id        = Column(Integer, ForeignKey("operator.operator_id"), nullable=True)
    occurred_at        = Column(DateTime(timezone=True))
    severity           = Column(SmallInteger)
    shift              = Column(String)
    comment            = Column(String)
    is_label_event     = Column(Boolean, nullable=False, server_default="false")


class GoldMachineHourlyFeature(Base):
    __tablename__ = "gold_machine_hourly_feature"

    feature_row_id                 = Column(Integer, primary_key=True, autoincrement=True)
    ingestion_batch_id             = Column(UUID(as_uuid=True),
                                            ForeignKey("ingestion_batch.ingestion_batch_id"))
    machine_id                     = Column(String)
    window_start                   = Column(DateTime(timezone=True))
    window_end                     = Column(DateTime(timezone=True))
    temp_mean_24h                  = Column(Float)
    temp_max_24h                   = Column(Float)
    temp_std_24h                   = Column(Float)
    pressure_mean_24h              = Column(Float)
    pressure_std_24h               = Column(Float)
    incident_count_prev_24h        = Column(Integer)
    incident_max_severity_prev_24h = Column(Integer)
    label_failure_next_24h         = Column(Boolean)
    split_set                      = Column(String)  # train | validation | test
