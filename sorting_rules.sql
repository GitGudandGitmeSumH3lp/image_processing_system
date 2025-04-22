CREATE TABLE sorting_rules (
    id SERIAL PRIMARY KEY, -- Unique identifier for each rule
    address_pattern VARCHAR(255) UNIQUE NOT NULL, -- The text/pattern to look for (e.g., '123 Main St', 'North Wing')
    sorting_destination VARCHAR(255) NOT NULL, -- Where this address should be sorted (e.g., 'Bin A', 'Servo Channel 1', 'Warehouse Section 3')
    priority INTEGER DEFAULT 0, -- Optional: for rules with overlapping patterns, higher priority wins
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Optional: Add an index for faster lookups if you have many rules
CREATE INDEX idx_address_pattern ON sorting_rules (address_pattern);