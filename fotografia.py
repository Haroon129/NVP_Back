class Foto:
    def __init__(self, id = None, url=None, size=None, label=None, predicted_label=None, probabilities = None):
        self.__id = id
        self.__url = url
        self.__size = size
        self.__label = label
        self.__predicted_label = predicted_label
        self.__probabilities = probabilities

# ID
    def set_id(self, id):
        self.__id = id
    
    def get_id(self):
        return self.__id

# URL
    def set_url(self, url):
        self.__url = url
    
    def get_url(self):
        return self.__url

# Size

    def set_size(self, size):
        self.__size = size

    def get_size(self):
        return self.__size
    
# Label
    def set_label(self, label):
        self.__label = label
    
    def get_label(self):
        return self.__label
    
# Predicted_label
    def set_predicted_label(self, label):
        self.__predicted_label = label
    
    def get_predicted_label(self):
        return self.__predicted_label
    
# Probabilities (NUEVOS MÉTODOS)
    def set_probabilities(self, probabilities):
        """Almacena la lista de probabilidades de todas las clases."""
        self.__probabilities = probabilities

    def get_probabilities(self):
        return self.__probabilities

# Str
    def __str__(self):
        # Opcional: puedes incluir la probabilidad en la salida __str__
        return  f"Foto ID {self.__id}:\n"\
                f"  url={self.__url}\n"\
                f"  size={self.__size}\n"\
                f"  label={self.__label}\n"\
                f"  predicted={self.__predicted_label}\n"\
                f"  probabilities={self.__probabilities}\n"