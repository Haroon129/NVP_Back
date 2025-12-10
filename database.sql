-- Script de creación de la base de datos y la tabla

-- 1. Creación de la base de datos NVP
CREATE DATABASE IF NOT EXISTS NVP;
USE NVP;

-- 2. Creación de la tabla imagenes
CREATE TABLE IF NOT EXISTS imagenes (
    id INT AUTO_INCREMENT PRIMARY KEY, -- Clave primaria auto incremental
    URL VARCHAR(255) NOT NULL,         -- Dirección de la imagen (hasta 255 caracteres)
    predicted_label INT NOT NULL,      -- Etiqueta predicha (un número entero)
    label INT NOT NULL                 -- Etiqueta real (otro número entero)
);
    