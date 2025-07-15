"""
IoT Botnet Test Dataset Generator
Creates synthetic test datasets in UNSW 2018 IoT Botnet format with different attack distributions
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime
import os

# Ensure testing_datasets directory exists
os.makedirs('testing_datasets', exist_ok=True)

def generate_base_features():
    """Generate realistic base network features"""
    return {
        'pkts': np.random.randint(1, 100),
        'bytes': np.random.randint(40, 10000),
        'seq': np.random.randint(1000, 999999),
        'dur': round(np.random.uniform(0.001, 5.0), 6),
        'mean': round(np.random.uniform(0.001, 2.0), 6),
        'stddev': round(np.random.uniform(0.0, 1.0), 6),
        'res_bps_payload': round(np.random.uniform(0, 1000), 2),
        'res_pps_payload': round(np.random.uniform(0, 100), 2),
        'res_bps_ratio': round(np.random.uniform(0, 1), 6),
        'res_pps_ratio': round(np.random.uniform(0, 1), 6),
        'ar_bps_payload': round(np.random.uniform(0, 1000), 2),
        'ar_pps_payload': round(np.random.uniform(0, 100), 2),
        'ar_bps_ratio': round(np.random.uniform(0, 1), 6),
        'ar_pps_ratio': round(np.random.uniform(0, 1), 6),
        'ar_bps_delta': round(np.random.uniform(0, 500), 2),
        'trans_depth': np.random.randint(0, 10),
        'response_body_len': np.random.randint(0, 5000),
        'ct_srv_src': np.random.randint(1, 50),
        'ct_srv_dst': np.random.randint(1, 50),
        'ct_dst_ltm': np.random.randint(1, 100),
        'ct_src_ltm': np.random.randint(1, 100),
        'ct_src_dport_ltm': np.random.randint(0, 20),
        'ct_dst_sport_ltm': np.random.randint(0, 20)
    }

def generate_normal_traffic():
    """Generate normal network traffic patterns"""
    base = generate_base_features()
    
    # Normal traffic characteristics
    base.update({
        'pkts': np.random.randint(1, 20),  # Lower packet counts
        'bytes': np.random.randint(40, 1500),  # Typical packet sizes
        'dur': round(np.random.uniform(0.001, 0.5), 6),  # Shorter durations
        'mean': round(np.random.uniform(0.001, 0.1), 6),  # Lower means
        'proto': random.choice(['tcp', 'udp', 'icmp']),
        'flgs': random.choice(['e', 'F', 'S', 'SA']),
        'state': random.choice(['INT', 'FIN', 'CON', 'REQ']),
        'category': 'Normal',
        'subcategory': 'Normal'
    })
    
    return base

def generate_ddos_traffic():
    """Generate DDoS attack patterns"""
    base = generate_base_features()
    
    # DDoS characteristics - high volume, short duration
    base.update({
        'pkts': np.random.randint(50, 500),  # High packet counts
        'bytes': np.random.randint(100, 2000),
        'dur': round(np.random.uniform(0.001, 0.1), 6),  # Very short
        'mean': round(np.random.uniform(0.001, 0.05), 6),
        'proto': random.choice(['tcp', 'udp']),
        'flgs': random.choice(['S', 'F', 'R']),
        'state': random.choice(['INT', 'CON']),
        'ct_srv_dst': np.random.randint(50, 200),  # High connection counts
        'category': 'DDoS',
        'subcategory': random.choice(['HTTP_Flood', 'UDP_Flood', 'TCP_Flood'])
    })
    
    return base

def generate_dos_traffic():
    """Generate DoS attack patterns"""
    base = generate_base_features()
    
    # DoS characteristics
    base.update({
        'pkts': np.random.randint(20, 100),
        'bytes': np.random.randint(100, 3000),
        'dur': round(np.random.uniform(0.1, 2.0), 6),
        'proto': random.choice(['tcp', 'udp']),
        'flgs': random.choice(['S', 'F', 'R', 'SA']),
        'state': random.choice(['INT', 'CON', 'FIN']),
        'category': 'DoS',
        'subcategory': random.choice(['TCP_SYN', 'UDP_Flood', 'ICMP_Flood'])
    })
    
    return base

def generate_reconnaissance_traffic():
    """Generate reconnaissance/scanning patterns"""
    base = generate_base_features()
    
    # Recon characteristics - many small probes
    base.update({
        'pkts': np.random.randint(1, 10),  # Small packets
        'bytes': np.random.randint(40, 200),  # Minimal data
        'dur': round(np.random.uniform(0.001, 0.1), 6),
        'proto': random.choice(['tcp', 'udp', 'icmp']),
        'flgs': random.choice(['S', 'F', 'R']),
        'state': random.choice(['INT', 'REQ']),
        'ct_srv_dst': np.random.randint(1, 5),  # Low connections
        'category': 'Reconnaissance',
        'subcategory': random.choice(['Service_Scan', 'Port_Scan', 'OS_Fingerprint'])
    })
    
    return base

def generate_theft_traffic():
    """Generate data theft patterns"""
    base = generate_base_features()
    
    # Data theft characteristics
    base.update({
        'pkts': np.random.randint(30, 200),
        'bytes': np.random.randint(1000, 50000),  # Large data transfers
        'dur': round(np.random.uniform(1.0, 10.0), 6),  # Longer sessions
        'proto': 'tcp',
        'flgs': random.choice(['SA', 'FA', 'F']),
        'state': random.choice(['CON', 'FIN']),
        'response_body_len': np.random.randint(1000, 20000),
        'category': 'Theft',
        'subcategory': random.choice(['Data_Exfiltration', 'Keylogger', 'Backdoor'])
    })
    
    return base

def create_network_row(row_id, timestamp, traffic_data):
    """Create a complete network traffic row"""
    return [
        row_id,  # pkSeqID
        timestamp,  # stime
        traffic_data.get('flgs', 'e'),  # flgs
        traffic_data.get('proto', 'tcp'),  # proto
        f"192.168.100.{np.random.randint(1, 254)}",  # saddr
        np.random.randint(1024, 65535),  # sport
        f"192.168.100.{np.random.randint(1, 254)}",  # daddr
        np.random.randint(80, 65535),  # dport
        traffic_data['pkts'],  # pkts
        traffic_data['bytes'],  # bytes
        traffic_data.get('state', 'INT'),  # state
        timestamp + traffic_data['dur'],  # ltime
        traffic_data['seq'],  # seq
        traffic_data['dur'],  # dur
        traffic_data['mean'],  # mean
        traffic_data['stddev'],  # stddev
        traffic_data.get('res_bps_payload', ''),  # res_bps_payload
        traffic_data.get('res_pps_payload', ''),  # res_pps_payload
        traffic_data['res_bps_ratio'],  # res_bps_ratio
        traffic_data['res_pps_ratio'],  # res_pps_ratio
        traffic_data.get('ar_bps_payload', ''),  # ar_bps_payload
        traffic_data.get('ar_pps_payload', ''),  # ar_pps_payload
        traffic_data['ar_bps_ratio'],  # ar_bps_ratio
        traffic_data['ar_pps_ratio'],  # ar_pps_ratio
        traffic_data.get('ar_bps_delta', ''),  # ar_bps_delta
        traffic_data['trans_depth'],  # trans_depth
        traffic_data['response_body_len'],  # response_body_len
        traffic_data['ct_srv_src'],  # ct_srv_src
        traffic_data['ct_srv_dst'],  # ct_srv_dst
        traffic_data['ct_dst_ltm'],  # ct_dst_ltm
        traffic_data['ct_src_ltm'],  # ct_src_ltm
        traffic_data['ct_src_dport_ltm'],  # ct_src_dport_ltm
        traffic_data['ct_dst_sport_ltm'],  # ct_dst_sport_ltm
        traffic_data['category'],  # category
        traffic_data['subcategory']  # subcategory
    ]

def generate_dataset(filename, normal_count, attack_distributions, total_size=5000):
    """Generate a complete test dataset"""
    
    print(f"Generating {filename}...")
    
    rows = []
    row_id = 1000001
    base_timestamp = 1526949252.645504
    
    # Generate normal traffic
    for i in range(normal_count):
        traffic = generate_normal_traffic()
        timestamp = base_timestamp + (i * 0.001)
        row = create_network_row(row_id, timestamp, traffic)
        rows.append(row)
        row_id += 1
    
    # Generate attacks based on distribution
    attack_count = total_size - normal_count
    current_offset = normal_count
    
    for attack_type, percentage in attack_distributions.items():
        count = int(attack_count * percentage)
        
        for i in range(count):
            if attack_type == 'DDoS':
                traffic = generate_ddos_traffic()
            elif attack_type == 'DoS':
                traffic = generate_dos_traffic()
            elif attack_type == 'Reconnaissance':
                traffic = generate_reconnaissance_traffic()
            elif attack_type == 'Theft':
                traffic = generate_theft_traffic()
            
            timestamp = base_timestamp + ((current_offset + i) * 0.001)
            row = create_network_row(row_id, timestamp, traffic)
            rows.append(row)
            row_id += 1
        
        current_offset += count
    
    # Shuffle the data to mix normal and attack traffic
    random.shuffle(rows)
    
    # Write to CSV
    with open(f'testing_datasets/{filename}', 'w') as f:
        for row in rows:
            # Convert row to string with proper formatting
            row_str = ','.join([str(item) if item != '' else '""' for item in row])
            f.write(row_str + '\n')
    
    # Calculate actual distribution
    normal_pct = (normal_count / total_size) * 100
    attack_pct = 100 - normal_pct
    
    # Show attack breakdown
    for attack_type, percentage in attack_distributions.items():
        count = int((total_size - normal_count) * percentage)
        print(f"      - {attack_type}: {count:,}")
    print()

# Generate different test datasets
if __name__ == "__main__":
    print("Generating IoT Botnet Test Datasets...")
    print("=" * 50)
    
    # Dataset 1: Mostly Normal Traffic (Good for testing false positives)
    generate_dataset(
        filename="mostly_normal_traffic.csv",
        normal_count=4000,
        attack_distributions={
            'Reconnaissance': 0.6,  # 60% of attacks are recon
            'DoS': 0.3,            # 30% DoS
            'DDoS': 0.1            # 10% DDoS
        },
        total_size=5000
    )
    
    # Dataset 2: Balanced Traffic (Good baseline test)
    generate_dataset(
        filename="balanced_network_traffic.csv",
        normal_count=2500,
        attack_distributions={
            'DDoS': 0.4,           # 40% DDoS
            'DoS': 0.3,            # 30% DoS
            'Reconnaissance': 0.2,  # 20% Recon
            'Theft': 0.1           # 10% Theft
        },
        total_size=5000
    )
    
    # Dataset 3: Mixed Attack Types (Good for testing different attack detection)
    generate_dataset(
        filename="mixed_attack_scenarios.csv",
        normal_count=1500,
        attack_distributions={
            'DDoS': 0.25,          # 25% each type
            'DoS': 0.25,
            'Reconnaissance': 0.25,
            'Theft': 0.25
        },
        total_size=5000
    )
    
    # Dataset 4: Stealth Attacks (Harder to detect)
    generate_dataset(
        filename="stealth_attack_traffic.csv",
        normal_count=3500,
        attack_distributions={
            'Reconnaissance': 0.7,  # Mostly scanning
            'Theft': 0.3           # Some data theft
        },
        total_size=5000
    )
    
    # Dataset 5: High Volume Attack (DDoS heavy)
    generate_dataset(
        filename="ddos_heavy_traffic.csv",
        normal_count=1000,
        attack_distributions={
            'DDoS': 0.8,           # 80% DDoS
            'DoS': 0.2             # 20% DoS
        },
        total_size=5000
    )