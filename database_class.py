import mysql.connector

class DatabaseConnection:
    """
    A class to manage connection and interaction with a MySQL database.
    
    It implements the Context Manager protocol (__enter__ and __exit__) 
    to ensure the database connection is automatically opened and closed.
    
    Attributes:
        host (str): Database server address.
        user (str): Username for authentication.
        database (str): Name of the database.
        password (str): Password for authentication (default is '').
        cnx (mysql.connector.connection): The active database connection object.
        cursor (mysql.connector.cursor): The active cursor object.
    """
    def __init__(self, host, user, database, password=''):
        """
        Initializes the connection parameters.

        Args:
            host (str): The hostname or IP address of the MySQL server.
            user (str): The user name for connecting to the database.
            database (str): The name of the database to connect to.
            password (str, optional): The password for the specified user. Defaults to ''.
        """
        self.host = host
        self.user = user
        self.cnx = None
        self.cursor = None
        self.password = password
        self.database = database
        
    def get_cnx(self):
        """
        Establishes and returns a connection to the MySQL database.
        
        Returns:
            mysql.connector.connection or None: The connection object if successful, 
                                                or None if a connection error occurs.
        """
        try:
            self.cnx = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            return self.cnx
        except mysql.connector.Error as err:
            print(f"❌ Connection Error: {err}")
            return None
            
    def close_cnx(self):
        """
        Closes the active database connection if it exists and is open.
        """
        # Checks if the connection exists (is not None) and is open before closing.
        if self.cnx and self.cnx.is_connected():
            self.cnx.close()
            
    def __enter__(self):
        """
        Context Manager entry method. Establishes the database connection.

        Returns:
            DatabaseConnection: The instance of the class itself.
        """
        self.get_cnx()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context Manager exit method. Closes the database connection.
        
        Args:
            exc_type: Exception type if an exception occurred within the 'with' block.
            exc_val: Exception value if an exception occurred.
            exc_tb: Traceback object if an exception occurred.
        """
        self.close_cnx()
        
    def execute_query(self, sql, params=None):
        """
        Executes INSERT, UPDATE, or DELETE queries and performs a COMMIT.
        
        Args:
            sql (str): The SQL query string to be executed.
            params (tuple, optional): A tuple of parameters to substitute into the query. 
                                      Defaults to None.

        Returns:
            bool: True if the query was executed and committed successfully, False otherwise.
        """
        if not self.cnx or not self.cnx.is_connected():
            print("❌ Error: No active connection.")
            return False
        try:
            self.cursor = self.cnx.cursor()
            self.cursor.execute(sql, params or ())
            self.cnx.commit()
            print(f"✅ Query executed and committed. Rows affected: {self.cursor.rowcount}")
            return True
        except mysql.connector.Error as err:
            print(f"❌ Error executing query: {err}")
            self.cnx.rollback() # Rolls back changes if an error occurs
            return False
        finally:
            if self.cursor:
                self.cursor.close()

    def select_all(self, sql, params=None):
        """
        Executes a SELECT query and returns all resulting rows as dictionaries.
        
        Args:
            sql (str): The SQL query string (e.g., 'SELECT * FROM table_name WHERE condition').
            params (tuple, optional): A tuple of parameters to substitute into the query. 
                                      Defaults to None.

        Returns:
            list: A list of dictionaries representing the selected rows, or an empty list on error.
        """
        if not self.cnx or not self.cnx.is_connected():
            print("❌ Error: No active connection.")
            return []
            
        try:
            # dictionary=True fetches results as dictionaries for easy column access
            self.cursor = self.cnx.cursor(dictionary=True) 
            self.cursor.execute(sql, params or ())
            results = self.cursor.fetchall()
            return results
        except mysql.connector.Error as err:
            print(f"❌ Error executing SELECT: {err}")
            return []
        finally:
            if self.cursor:
                self.cursor.close()

def update_label_pred_incorrecta(self, target_url: str, new_label_value: int) -> bool:
    update_sql = """
    UPDATE imagenes
    SET label = %s
    WHERE URL = %s;
    """
    update_params = (int(new_label_value), target_url)

    update_success = self.execute_query(update_sql, update_params)
    if not update_success:
        print("❌ Operación abortada: El UPDATE falló.")
        return False
    return True


def update_label_pred_correcta(self, target_url: str) -> bool:
    SQL_SYNC = """
    UPDATE imagenes
    SET label = predicted_label
    WHERE URL = %s;
    """
    update_params = (target_url,)  # 👈 coma para que sea tupla

    update_success = self.execute_query(SQL_SYNC, update_params)
    if not update_success:
        print("❌ Operación abortada: El UPDATE falló.")
        return False
    return True

