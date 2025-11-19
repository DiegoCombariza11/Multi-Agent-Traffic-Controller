import xml.etree.ElementTree as ET
import random

# Configuración
INPUT_FILE = "./sumoData/osm.passenger.trips.xml"
OUTPUT_FILE = "./sumoData/osm.passenger.trips_lite.xml"
KEEP_PERCENTAGE = 0.2  

def reduce_traffic():
    print(f"Leyendo {INPUT_FILE}...")
    tree = ET.parse(INPUT_FILE)
    root = tree.getroot()
    
    original_count = len(root.findall('trip'))
    new_trips = []
    
    for trip in root.findall('trip'):
        if random.random() < KEEP_PERCENTAGE:
            new_trips.append(trip)
            
    # Limpiamos el root y ponemos solo los trips seleccionados
    root.clear()
    # Volvemos a añadir la configuración base si la hubiera (vtypes, etc)
    # Para trips simple, basta con re-insertar los trips
    for trip in new_trips:
        root.append(trip)
        
    print(f"Reducción completada: {original_count} -> {len(new_trips)} vehículos.")
    tree.write(OUTPUT_FILE)
    print(f"Guardado en: {OUTPUT_FILE}")
    print("¡AHORA RECUERDA CAMBIAR TU CONFIG.PY PARA USAR ESTE ARCHIVO!")

if __name__ == "__main__":
    reduce_traffic()