from sqlalchemy import Column, String, Float, Integer, Boolean
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class BronzeTelemetry(Base):
    __tablename__ = "bronze_telemetry"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    machine_id        = Column(String)
    timestamp         = Column(String)
    temperature_c     = Column(Float)
    pressure_bar      = Column(Float)
    voltage_mean_v    = Column(Float)
    rotation_mean_rpm = Column(Float)
    pieces_produced   = Column(Integer)
    parse_ok          = Column(Boolean, nullable=False, server_default="true")
    parse_ok_reason   = Column(String,  nullable=False, server_default="")


class BronzeIncident(Base):
    __tablename__ = "bronze_incidents"

    id                      = Column(Integer, primary_key=True, autoincrement=True)
    incident_id             = Column(String)
    date                    = Column(String)
    time                    = Column(String)
    operator_name           = Column(String)
    machine_id              = Column(String)
    severity                = Column(Integer)
    operator_badge          = Column(String)
    comment                 = Column(String)
    shift                   = Column(String)
    type_surchauffe         = Column(Integer)
    type_baisse_pression    = Column(Integer)
    type_vibration          = Column(Integer)
    type_bruit_mecanique    = Column(Integer)
    type_surconsommation    = Column(Integer)
    type_blocage_mecanique  = Column(Integer)
    type_alarme_capteur     = Column(Integer)
    type_arret_urgence      = Column(Integer)
    type_defaut_qualite     = Column(Integer)
    parse_ok                = Column(Boolean, nullable=False, server_default="true")
    parse_ok_reason         = Column(String,  nullable=False, server_default="")
