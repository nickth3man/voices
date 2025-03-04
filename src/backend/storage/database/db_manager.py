"""
Database manager for the Voices application.

This module provides a DatabaseManager class for managing database connections,
creating tables, and performing common database operations.
"""

import os
import logging
import datetime
from typing import Optional, List, Dict, Any, Type, Union

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from .models import Base, Speaker, ProcessedFile, ProcessingHistory, ModelPerformance
from .migrations import run_migrations

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manager for database operations.
    
    This class provides methods for connecting to the database, creating tables,
    and performing common database operations.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the DatabaseManager.
        
        Args:
            db_path: Path to the SQLite database file. If None, uses a default path.
        """
        if db_path is None:
            # Use default path in user's home directory
            home_dir = os.path.expanduser("~")
            app_dir = os.path.join(home_dir, ".voices")
            os.makedirs(app_dir, exist_ok=True)
            db_path = os.path.join(app_dir, "voices.db")
        
        self.db_path = db_path
        self.engine = None
        self.Session = None
        self._session = None
    
    def connect(self) -> bool:
        """
        Connect to the database.
        
        Returns:
            bool: True if connection successful, False otherwise.
        """
        try:
            # Create SQLite database engine
            self.engine = create_engine(f"sqlite:///{self.db_path}")
            self.Session = sessionmaker(bind=self.engine)
            logger.info(f"Connected to database at {self.db_path}")
            return True
        except SQLAlchemyError as e:
            logger.error(f"Error connecting to database: {str(e)}")
            return False
    
    def create_tables(self) -> bool:
        """
        Create database tables if they don't exist.
        
        Returns:
            bool: True if tables created successfully, False otherwise.
        """
        if self.engine is None:
            logger.error("Cannot create tables: Not connected to database")
            return False
        
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Database tables created successfully")
            return True
        except SQLAlchemyError as e:
            logger.error(f"Error creating database tables: {str(e)}")
            return False
    
    def run_migrations(self) -> bool:
        """
        Run database migrations.
        
        Returns:
            bool: True if migrations ran successfully, False otherwise.
        """
        if self.engine is None:
            logger.error("Cannot run migrations: Not connected to database")
            return False
        
        try:
            run_migrations(self.engine)
            logger.info("Database migrations completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error running database migrations: {str(e)}")
            return False
    
    def initialize(self) -> bool:
        """
        Initialize the database by connecting, creating tables, and running migrations.
        
        Returns:
            bool: True if initialization successful, False otherwise.
        """
        if not self.connect():
            return False
        
        if not self.create_tables():
            return False
        
        if not self.run_migrations():
            return False
        
        return True
    
    @property
    def session(self) -> Session:
        """
        Get a database session.
        
        Returns:
            Session: SQLAlchemy session object.
        """
        if self._session is None or not self._session.is_active:
            if self.Session is None:
                raise RuntimeError("Cannot create session: Not connected to database")
            self._session = self.Session()
        
        return self._session
    
    def close_session(self):
        """Close the current database session."""
        if self._session is not None:
            self._session.close()
            self._session = None
    
    def add(self, obj: Any) -> bool:
        """
        Add an object to the database.
        
        Args:
            obj: SQLAlchemy model object to add.
            
        Returns:
            bool: True if added successfully, False otherwise.
        """
        try:
            self.session.add(obj)
            self.session.commit()
            return True
        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"Error adding object to database: {str(e)}")
            return False
    
    def add_all(self, objects: List[Any]) -> bool:
        """
        Add multiple objects to the database.
        
        Args:
            objects: List of SQLAlchemy model objects to add.
            
        Returns:
            bool: True if added successfully, False otherwise.
        """
        try:
            self.session.add_all(objects)
            self.session.commit()
            return True
        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"Error adding objects to database: {str(e)}")
            return False
    
    def update(self, obj: Any) -> bool:
        """
        Update an object in the database.
        
        Args:
            obj: SQLAlchemy model object to update.
            
        Returns:
            bool: True if updated successfully, False otherwise.
        """
        try:
            self.session.commit()
            return True
        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"Error updating object in database: {str(e)}")
            return False
    
    def delete(self, obj: Any) -> bool:
        """
        Delete an object from the database.
        
        Args:
            obj: SQLAlchemy model object to delete.
            
        Returns:
            bool: True if deleted successfully, False otherwise.
        """
        try:
            self.session.delete(obj)
            self.session.commit()
            return True
        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"Error deleting object from database: {str(e)}")
            return False
    
    def get_by_id(self, model_class: Type, obj_id: int) -> Optional[Any]:
        """
        Get an object by its ID.
        
        Args:
            model_class: SQLAlchemy model class.
            obj_id: ID of the object to get.
            
        Returns:
            Optional[Any]: The object if found, None otherwise.
        """
        try:
            return self.session.query(model_class).get(obj_id)
        except SQLAlchemyError as e:
            logger.error(f"Error getting object by ID: {str(e)}")
            return None
    
    def get_all(self, model_class: Type) -> List[Any]:
        """
        Get all objects of a specific model class.
        
        Args:
            model_class: SQLAlchemy model class.
            
        Returns:
            List[Any]: List of objects.
        """
        try:
            return self.session.query(model_class).all()
        except SQLAlchemyError as e:
            logger.error(f"Error getting all objects: {str(e)}")
            return []
    
    def query(self, model_class: Type) -> Any:
        """
        Get a query object for a specific model class.
        
        Args:
            model_class: SQLAlchemy model class.
            
        Returns:
            Any: SQLAlchemy query object.
        """
        return self.session.query(model_class)
    
    def execute_raw_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a raw SQL query.
        
        Args:
            query: SQL query string.
            params: Query parameters.
            
        Returns:
            List[Dict[str, Any]]: Query results as a list of dictionaries.
        """
        try:
            result = self.engine.execute(query, params or {})
            return [dict(row) for row in result]
        except SQLAlchemyError as e:
            logger.error(f"Error executing raw query: {str(e)}")
            return []
    
    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.
        
        Args:
            table_name: Name of the table to check.
            
        Returns:
            bool: True if the table exists, False otherwise.
        """
        if self.engine is None:
            logger.error("Cannot check table existence: Not connected to database")
            return False
        
        inspector = inspect(self.engine)
        return table_name in inspector.get_table_names()
    
    def get_table_columns(self, table_name: str) -> List[Dict[str, Any]]:
        """
        Get information about columns in a table.
        
        Args:
            table_name: Name of the table.
            
        Returns:
            List[Dict[str, Any]]: List of column information dictionaries.
        """
        if self.engine is None:
            logger.error("Cannot get table columns: Not connected to database")
            return []
        
        inspector = inspect(self.engine)
        return inspector.get_columns(table_name)
    
    def backup_database(self, backup_path: Optional[str] = None) -> bool:
        """
        Create a backup of the database.
        
        Args:
            backup_path: Path for the backup file. If None, uses default path.
            
        Returns:
            bool: True if backup successful, False otherwise.
        """
        if backup_path is None:
            # Use default backup path
            backup_dir = os.path.dirname(self.db_path)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(backup_dir, f"voices_backup_{timestamp}.db")
        
        try:
            # For SQLite, we can simply copy the database file
            import shutil
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Database backed up to {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Error backing up database: {str(e)}")
            return False