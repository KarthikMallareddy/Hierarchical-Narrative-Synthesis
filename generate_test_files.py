import os
import csv
import random

os.makedirs("test_files", exist_ok=True)

# 1. Generate a CSV file (Financial Data)
with open("test_files/q3_financial_report.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Date", "Department", "Revenue", "Platform_Costs"])
    for i in range(1, 15):
        writer.writerow([f"2023-09-{i:02d}", random.choice(["Sales", "Engineering", "Marketing"]), 
                         round(random.uniform(5000, 20000), 2), round(random.uniform(1000, 5000), 2)])
print("✅ Created test_files/q3_financial_report.csv")

# 2. Generate a LOG file (Server Events)
with open("test_files/server_events.log", "w") as f:
    f.write("2023-09-01 10:00:01 INFO [main] Server started on port 8080\n")
    f.write("2023-09-01 10:05:12 WARN [db] Connection pool reaching limit\n")
    f.write("2023-09-01 10:45:33 ERROR [api] Timeout while fetching user data\n")
    f.write("2023-09-01 11:12:00 INFO [main] Memory utilization at 45%\n")
    f.write("2023-09-02 08:30:15 ERROR [db] Connection lost. Retrying...\n")
print("✅ Created test_files/server_events.log")

# 3. Generate a TXT file (Research Abstract)
with open("test_files/research_memo.txt", "w") as f:
    f.write("Internal Memo: Next-Gen Architecture Plan\n\n")
    f.write("We are planning to transition our backend to a serverless architecture to reduce platform costs. ")
    f.write("Initial tests show a 30% reduction in latency for API requests, although resolving database connection limits remains a priority. ")
    f.write("Expected completion for Phase 1 is Q4 2023.\n")
print("✅ Created test_files/research_memo.txt")

print("\n🎉 Test files are ready! You can now upload these into the Streamlit app sidebar.")
