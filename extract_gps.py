#!/usr/bin/env python3
"""
Extract GPS coordinates from XML metadata files and create COLMAP-compatible GPS file.
"""

import os
import sys
import argparse
import glob
import xml.etree.ElementTree as ET

def get_geotagging(exif):
    """Extract GPS information from EXIF data."""
    if not exif:
        return None
    
    geotagging = {}
    for key, val in TAGS.items():
        if val == "GPSInfo":
            if key in exif:
                for gps_key, gps_val in GPSTAGS.items():
                    if gps_key in exif[key]:
                        geotagging[gps_val] = exif[key][gps_key]
            break
    
    return geotagging if geotagging else None

def get_decimal_from_dms(dms, ref):
    """Convert DMS (degrees, minutes, seconds) to decimal degrees."""
    degrees = dms[0]
    minutes = dms[1] / 60.0
    seconds = dms[2] / 3600.0
    
    if ref in ['S', 'W']:
        degrees = -degrees
        minutes = -minutes  
        seconds = -seconds
    
    return degrees + minutes + seconds

def get_coordinates(geotagging):
    """Extract latitude and longitude from GPS data."""
    lat = geotagging.get('GPSLatitude')
    lat_ref = geotagging.get('GPSLatitudeRef')
    lon = geotagging.get('GPSLongitude')
    lon_ref = geotagging.get('GPSLongitudeRef')
    alt = geotagging.get('GPSAltitude', 0)
    
    if lat and lat_ref and lon and lon_ref:
        lat_decimal = get_decimal_from_dms(lat, lat_ref)
        lon_decimal = get_decimal_from_dms(lon, lon_ref)
        
        # Handle altitude
        if alt:
            if isinstance(alt, tuple):
                alt_decimal = float(alt[0]) / float(alt[1]) if alt[1] != 0 else 0
            else:
                alt_decimal = float(alt)
        else:
            alt_decimal = 0
            
        return lat_decimal, lon_decimal, alt_decimal
    
    return None, None, None

def parse_xml_metadata(xml_file):
    """Parse GPS and camera info from XML file."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Extract GPS coordinates
        gps_lat = root.find('GPS_LATITUDE')
        gps_lon = root.find('GPS_LONGITUDE') 
        gps_alt = root.find('GPS_ALTITUDE')
        
        # Extract Korean coordinate system (더 정확한 좌표)
        coord_x = root.find('.//원점X좌표')
        coord_y = root.find('.//원점Y좌표')
        coord_z = root.find('.//원점Z좌표')
        
        # Camera orientation
        omega = root.find('.//Omega')
        phi = root.find('.//Phi')
        kappa = root.find('.//Kappa')
        
        result = {}
        
        # Use Korean coordinate system if available (more accurate)
        if coord_x is not None and coord_y is not None and coord_z is not None:
            result['x'] = float(coord_x.text)
            result['y'] = float(coord_y.text)
            result['z'] = float(coord_z.text)
            result['coordinate_system'] = 'TM'  # Transverse Mercator
        # Fallback to GPS
        elif gps_lat is not None and gps_lon is not None:
            # Parse GPS format (N37.468947 -> 37.468947)
            lat_str = gps_lat.text.replace('N', '').replace('S', '-')
            lon_str = gps_lon.text.replace('E', '').replace('W', '-')
            
            result['x'] = float(lon_str)  # Longitude as X
            result['y'] = float(lat_str)  # Latitude as Y
            result['z'] = float(gps_alt.text) if gps_alt is not None else 0
            result['coordinate_system'] = 'GPS'
        
        # Add orientation if available
        if omega is not None and phi is not None and kappa is not None:
            result['omega'] = float(omega.text)
            result['phi'] = float(phi.text) 
            result['kappa'] = float(kappa.text)
            
        return result
        
    except Exception as e:
        print(f"Error parsing XML {xml_file}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Extract GPS coordinates from XML metadata')
    parser.add_argument('--image_path', required=True, help='Path to images')
    parser.add_argument('--xml_path', help='Path to XML metadata files')
    parser.add_argument('--output', required=True, help='Output GPS file')
    args = parser.parse_args()
    
    # Auto-detect XML path if not provided
    if not args.xml_path:
        image_dir = os.path.basename(args.image_path.rstrip('/'))
        parent_dir = os.path.dirname(args.image_path)
        args.xml_path = os.path.join(parent_dir, image_dir + '_xml')
    
    print(f"Looking for XML metadata in: {args.xml_path}")
    
    # Find XML files
    xml_files = glob.glob(os.path.join(args.xml_path, "*.xml"))
    
    if not xml_files:
        print(f"No XML metadata files found in {args.xml_path}")
        return
    
    print(f"Found {len(xml_files)} XML metadata files")
    
    gps_data = []
    
    for xml_file in xml_files:
        xml_name = os.path.basename(xml_file)
        image_name = xml_name.replace('.xml', '.tif')  # Assume TIF images
        
        metadata = parse_xml_metadata(xml_file)
        if metadata and 'x' in metadata and 'y' in metadata:
            # COLMAP format: IMAGE_NAME X Y Z
            x, y, z = metadata['x'], metadata['y'], metadata['z'] 
            coord_sys = metadata.get('coordinate_system', 'Unknown')
            
            gps_data.append(f"{image_name} {x:.6f} {y:.6f} {z:.6f}")
            print(f"  {image_name}: ({x:.6f}, {y:.6f}, {z:.6f}) [{coord_sys}]")
    
    print(f"\nExtracted coordinates from {len(gps_data)} images")
    
    if gps_data:
        with open(args.output, 'w') as f:
            f.write("# GPS coordinates extracted from XML metadata\n")
            f.write("# Format: IMAGE_NAME X Y Z\n")
            f.write("# Coordinate system: TM (Transverse Mercator) or GPS\n")
            for line in gps_data:
                f.write(line + "\n")
        
        print(f"GPS coordinates saved to {args.output}")
    else:
        print("No GPS coordinates extracted")

if __name__ == "__main__":
    main()