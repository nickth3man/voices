"""
Database migrations for the Voices application.

This module provides functionality for managing database schema migrations,
allowing for schema updates over time.
"""

import os
import json
import logging
import datetime
from typing import Dict, Any, List, Optional

from sqlalchemy import create_engine, inspect, MetaData, Table, Column, Integer, String, DateTime
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

# Table to track migrations
MIGRATIONS_TABLE = "schema_migrations"


def get_migrations_dir() -> str:
    """
    Get the directory where migration files are stored.
    
    Returns:
        str: Path to migrations directory.
    """
    # Migrations are stored in a 'migrations' subdirectory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    migrations_dir = os.path.join(current_dir, "migrations")
    os.makedirs(migrations_dir, exist_ok=True)
    return migrations_dir


def ensure_migrations_table(engine: Engine) -> bool:
    """
    Ensure the migrations tracking table exists.
    
    Args:
        engine: SQLAlchemy engine.
        
    Returns:
        bool: True if table exists or was created, False on error.
    """
    metadata = MetaData()
    
    # Define the migrations table if it doesn't exist
    migrations_table = Table(
        MIGRATIONS_TABLE,
        metadata,
        Column("id", Integer, primary_key=True),
        Column("version", String(50), nullable=False, unique=True),
        Column("applied_at", DateTime, default=datetime.datetime.utcnow),
        Column("description", String(255), nullable=True)
    )
    
    try:
        # Create the table if it doesn't exist
        inspector = inspect(engine)
        if MIGRATIONS_TABLE not in inspector.get_table_names():
            migrations_table.create(engine)
            logger.info(f"Created {MIGRATIONS_TABLE} table")
        return True
    except SQLAlchemyError as e:
        logger.error(f"Error creating migrations table: {str(e)}")
        return False


def get_applied_migrations(engine: Engine) -> List[str]:
    """
    Get a list of migrations that have already been applied.
    
    Args:
        engine: SQLAlchemy engine.
        
    Returns:
        List[str]: List of applied migration versions.
    """
    try:
        # Ensure migrations table exists
        if not ensure_migrations_table(engine):
            return []
        
        # Query for applied migrations
        result = engine.execute(f"SELECT version FROM {MIGRATIONS_TABLE} ORDER BY version")
        return [row[0] for row in result]
    except SQLAlchemyError as e:
        logger.error(f"Error getting applied migrations: {str(e)}")
        return []


def get_available_migrations() -> List[Dict[str, Any]]:
    """
    Get a list of available migrations from the migrations directory.
    
    Returns:
        List[Dict[str, Any]]: List of migration dictionaries with version and path.
    """
    migrations_dir = get_migrations_dir()
    migrations = []
    
    try:
        # Look for .sql and .json migration files
        for filename in os.listdir(migrations_dir):
            if filename.endswith(".sql") or filename.endswith(".json"):
                # Extract version from filename (e.g., "001_initial_schema.sql" -> "001")
                version = filename.split("_")[0]
                path = os.path.join(migrations_dir, filename)
                migrations.append({
                    "version": version,
                    "path": path,
                    "filename": filename
                })
        
        # Sort by version
        migrations.sort(key=lambda m: m["version"])
        return migrations
    except Exception as e:
        logger.error(f"Error getting available migrations: {str(e)}")
        return []


def apply_migration(engine: Engine, migration: Dict[str, Any]) -> bool:
    """
    Apply a single migration.
    
    Args:
        engine: SQLAlchemy engine.
        migration: Migration dictionary with version and path.
        
    Returns:
        bool: True if migration was applied successfully, False otherwise.
    """
    version = migration["version"]
    path = migration["path"]
    filename = migration["filename"]
    
    try:
        # Read migration file
        with open(path, "r") as f:
            content = f.read()
        
        # Apply migration based on file type
        if path.endswith(".sql"):
            # SQL migration
            statements = content.split(";")
            for statement in statements:
                statement = statement.strip()
                if statement:
                    engine.execute(statement)
        elif path.endswith(".json"):
            # JSON migration (for more complex migrations)
            migration_data = json.loads(content)
            if "up" in migration_data:
                for statement in migration_data["up"]:
                    engine.execute(statement)
        
        # Record migration in migrations table
        description = filename.replace(f"{version}_", "").replace(".sql", "").replace(".json", "").replace("_", " ")
        engine.execute(
            f"INSERT INTO {MIGRATIONS_TABLE} (version, description) VALUES (?, ?)",
            (version, description)
        )
        
        logger.info(f"Applied migration {version}: {description}")
        return True
    except Exception as e:
        logger.error(f"Error applying migration {version}: {str(e)}")
        return False


def run_migrations(engine: Engine) -> bool:
    """
    Run all pending migrations.
    
    Args:
        engine: SQLAlchemy engine.
        
    Returns:
        bool: True if all migrations were applied successfully, False otherwise.
    """
    # Get applied and available migrations
    applied_migrations = get_applied_migrations(engine)
    available_migrations = get_available_migrations()
    
    # Determine pending migrations
    pending_migrations = [m for m in available_migrations if m["version"] not in applied_migrations]
    
    if not pending_migrations:
        logger.info("No pending migrations to apply")
        return True
    
    logger.info(f"Found {len(pending_migrations)} pending migrations")
    
    # Apply pending migrations in order
    success = True
    for migration in pending_migrations:
        if not apply_migration(engine, migration):
            success = False
            break
    
    return success


def create_migration(name: str, sql_content: Optional[str] = None) -> str:
    """
    Create a new migration file.
    
    Args:
        name: Name of the migration (e.g., "add_user_table").
        sql_content: Optional SQL content for the migration.
        
    Returns:
        str: Path to the created migration file.
    """
    migrations_dir = get_migrations_dir()
    
    # Get the next version number
    available_migrations = get_available_migrations()
    if available_migrations:
        last_version = int(available_migrations[-1]["version"])
        next_version = f"{last_version + 1:03d}"
    else:
        next_version = "001"
    
    # Create filename
    filename = f"{next_version}_{name.lower().replace(' ', '_')}.sql"
    path = os.path.join(migrations_dir, filename)
    
    # Write migration file
    with open(path, "w") as f:
        if sql_content:
            f.write(sql_content)
        else:
            f.write("-- Migration: {}\n-- Created: {}\n\n".format(
                name, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
    
    logger.info(f"Created migration file: {path}")
    return path


def create_initial_migration() -> str:
    """
    Create the initial migration for the database schema.
    
    Returns:
        str: Path to the created migration file.
    """
    # This migration is automatically created when the migrations module is first used
    # It contains the initial schema for the database
    
    sql_content = """-- Migration: Initial Schema
-- Created: {}

-- This migration creates the initial database schema for the Voices application.
-- It includes tables for speakers, processed files, processing history, and model performance metrics.

-- Note: The actual tables are created by SQLAlchemy's create_all() method,
-- so this migration is just a placeholder to mark the initial schema version.
""".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    return create_migration("initial_schema", sql_content)


# Create initial migration if it doesn't exist
if not os.path.exists(os.path.join(get_migrations_dir(), "001_initial_schema.sql")):
    create_initial_migration()