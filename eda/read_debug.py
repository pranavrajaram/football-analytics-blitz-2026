try:
    with open('debug_output_2.txt', encoding='utf-16') as f:
        for line in f:
            if "DEBUG:" in line:
               if "Original" in line:
                   print(line.strip())
               if "Mapped" in line:
                   # Parse dict string to just show keys or subset
                   print(line.strip()[:200]) # First 200 chars
except Exception as e:
    print(f"UTF-16 failed: {e}")
    try:
        with open('debug_output_2.txt', encoding='utf-8', errors='ignore') as f:
             for line in f:
                if "DEBUG:" in line:
                   if "Original" in line:
                       print(line.strip())
                   if "Mapped" in line:
                       print(line.strip()[:200])
    except Exception as e2:
        print(f"UTF-8 failed: {e2}")
