import csv

def export_to_csv(collection, csv_path):

    # Fetch all documents from the collection
    documents = collection.find()

    # Open a CSV file for writing
    with open(csv_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        
        # Write the header (field names)
        header = documents[0].keys()  # Get field names from the first document
        writer.writerow(header)
        
        # Write the data rows
        for doc in documents:
            writer.writerow([doc.get(field, "") for field in header])

    print(f"Exported collection to '{csv_path}' successfully.")