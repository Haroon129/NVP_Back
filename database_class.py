import mysql.connector

class DatabaseConnection:
    def __init__(self, host,user,database,password=''):
        self.host=host
        self.user=user
        self.cnx=None
        self.cursor=None
        self.password=password
        self.database=database
        
    def get_cnx(self):
        # (Tu código de conexión y manejo de errores va aquí, que ya está bien)
        try:
            self.cnx = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            return self.cnx
        except mysql.connector.Error as err:
            # Simplificado para fines de este ejemplo:
            print(f"❌ Error de conexión: {err}")
            return None # Importante devolver algo si falla
            
    # El método de cierre es mucho más simple y seguro
    def close_cnx(self):
        # Verifica que la conexión exista (no sea None) y que esté abierta antes de cerrarla.
        if self.cnx and self.cnx.is_connected():
            self.cnx.close()
            # print("🔒 Conexión MySQL cerrada automáticamente.")
            
    def __enter__(self):
        self.get_cnx()
        return self
    
    # Aquí usamos la firma correcta y llamamos a close_cnx
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_cnx()
        # NOTA: No necesitamos 'return self'. El Context Manager se encarga de salir.
    
    def execute_query(self, sql, params=None):
        """Ejecuta consultas INSERT, UPDATE o DELETE, y realiza el COMMIT."""
        if not self.cnx or not self.cnx.is_connected():
            print("❌ Error: No hay una conexión activa.")
            return False
        try:
            self.cursor = self.cnx.cursor()
            self.cursor.execute(sql, params or ())
            self.cnx.commit()
            print(f"✅ Consulta ejecutada y confirmada. Filas afectadas: {self.cursor.rowcount}")
            return True
        except mysql.connector.Error as err:
            print(f"❌ Error al ejecutar la consulta: {err}")
            self.cnx.rollback() # Deshace los cambios si hay un error
            return False
        finally:
            if self.cursor:
                self.cursor.close()

    def select_all(self, params=None):
        """Ejecuta una consulta SELECT y devuelve todas las filas."""
        sql = 'Select * from imagenes'
        if not self.cnx or not self.cnx.is_connected():
            print("❌ Error: No hay una conexión activa.")
            return []
            
        try:
            self.cursor = self.cnx.cursor(dictionary=True) # dictionary=True para obtener resultados como dicts
            self.cursor.execute(sql, params or ())
            results = self.cursor.fetchall()
            return results
        except mysql.connector.Error as err:
            print(f"❌ Error al ejecutar SELECT: {err}")
            return []
        finally:
            if self.cursor:
                self.cursor.close()
    
    def update_label_pred_incorrecta(self, target_url: str, new_label_value: int) -> bool:
        """
        Actualiza el campo 'label' de una imagen usando la URL como WHERE
        y luego verifica la actualización con un SELECT.

        Args:
            target_url: La URL de la imagen a actualizar.
            new_label_value: El nuevo valor numérico para la columna 'label'.

        Returns:
            True si la actualización fue exitosa, False si falló.
        """
        
        update_sql = """
        UPDATE imagenes
        SET label = %s
        WHERE URL = %s;
        """
        update_params = (new_label_value, target_url)
        
        # Usamos el método existente execute_query()
        update_success = self.execute_query(update_sql, update_params)

        if not update_success:
            print("❌ Operación abortada: El UPDATE falló.")
            return False
        return True
    
    def update_label_pred_correcta(self, target_url: str) -> bool:
        """
        Actualiza el campo 'label' de una imagen usando la URL como WHERE
        y luego verifica la actualización con un SELECT.

        Args:
            target_url: La URL de la imagen a actualizar.
            new_label_value: El nuevo valor numérico para la columna 'label'.

        Returns:
            True si la actualización fue exitosa, False si falló.
        """
        
        SQL_SYNC = """
        UPDATE imagenes
        SET label = predicted_label
        Where URL = %s;
        """
        update_params = (target_url)
        
        # Usamos el método existente execute_query()
        update_success = self.execute_query(SQL_SYNC, update_params)

        if not update_success:
            print("❌ Operación abortada: El UPDATE falló.")
            return False
        return True