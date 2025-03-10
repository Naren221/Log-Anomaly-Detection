import re
import csv

# Define event templates and their IDs
event_templates = [
    ("E1", r".*?:.*? Served block blk_.*? to /.*?"),
    ("E2", r".*?:.*? Starting thread to transfer block blk_.*? to .*?:.*?"),
    ("E3", r".*?:.*?:Got exception while serving blk_.*? to /.*?:"),
    ("E4", r"BLOCK\* ask .*?:.*? to delete blk_.*?"),
    ("E5", r"BLOCK\* ask .*?:.*? to replicate blk_.*? to datanode\(s\) .*?:.*?"),
    ("E6", r"BLOCK\* NameSystem\.addStoredBlock: blockMap updated: .*?:.*? is added to blk_.*? size .*?"),
    ("E7", r"BLOCK\* NameSystem\.allocateBlock: /.*?/part-.*?. blk_.*?"),
    ("E8", r"BLOCK\* NameSystem\.delete: blk_.*? is added to invalidSet of .*?:.*?"),
    ("E9", r"Deleting block blk_.*? file /.*?/blk_.*?"),
    ("E10", r"PacketResponder .*? for block blk_.*? terminating"),
    ("E11", r"Received block blk_.*? of size .*? from /.*?"),
    ("E12", r"Received block blk_.*? src: /.*?:.*? dest: /.*?:.*? of size .*?"),
    ("E13", r"Receiving block blk_.*? src: /.*?:.*? dest: /.*?:.*?"),
    ("E14", r"Verification succeeded for blk_.*?")
]

# Function to match a log line with event templates
def generate_event_id_and_template(content):
    """
    Matches a log content to the predefined event templates.
    Args:
        content (str): The log content to match.

    Returns:
        tuple: EventId and EventTemplate if matched, else "E_UNKNOWN" and the original content.
    """
    for event_id, template in event_templates:
        if re.search(template, content):
            # Substitute matched parts in content with template placeholders
            event_template = re.sub(template, template.replace(r".*?", "<*>"), content)
            return event_id, event_template
    return "E_UNKNOWN", content

# Function to parse a log line into structured fields
def parse_log_line(line):
    """
    Parses a log line into structured fields (Date, Time, Pid, Level, Component, Content).
    Args:
        line (str): The log line to parse.

    Returns:
        dict: A dictionary with structured log fields.
    """
    pattern = r"^(?P<date>\d+)\s+(?P<time>\d+)\s+(?P<pid>\d+)\s+(?P<level>[A-Z]+)\s+(?P<component>[^\s:]+):?\s+(?P<content>.+)$"
    match = re.match(pattern, line)
    if match:
        return match.groupdict()
    return None

# Function to process logs and write structured data to a CSV file
def process_logs(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        csv_writer = csv.writer(outfile)
        # Write header
        csv_writer.writerow(["LineId", "Date", "Time", "Pid", "Level", "Component", "Content", "EventId", "EventTemplate"])
        
        for line_id, line in enumerate(infile, start=1):
            line = line.strip()
            parsed_line = parse_log_line(line)
            if parsed_line:
                # Extract fields
                date = parsed_line['date']
                time = parsed_line['time']
                pid = parsed_line['pid']
                level = parsed_line['level']
                component = parsed_line['component']
                content = parsed_line['content']
                # Generate EventId and EventTemplate
                event_id, event_template = generate_event_id_and_template(content)
                # Write to CSV
                csv_writer.writerow([line_id, date, time, pid, level, component, content, event_id, event_template])

# Main execution
if __name__ == "__main__":
    # Define input log file and output CSV file paths
    log_file_path = "HDFS_2k.log"  # Input log file
    output_csv_path = "structured_logs.csv"  # Output CSV file

    # Process the logs and save structured data
    process_logs(log_file_path, output_csv_path)
