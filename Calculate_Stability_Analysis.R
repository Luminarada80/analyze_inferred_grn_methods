library(dplyr)
library(readr)
library(purrr)

sample_name <- "macrophage_stability_buffer4"
network_dir <- paste0("OUTPUT/LINGER/", sample_name, "/STANDARDIZED_INFERRED_NETWORKS")
output_path <- paste0("OUTPUT/LINGER/", sample_name, "/jaccard_index_stability_matrix.csv")

# Function to read, filter, and sort each network
read_and_sort_network <- function(file) {
  network <- read_csv(file)
  
  set.seed(12345) # Sets a seed for selecting edges so that the same ones are selected

  # Rename columns
  colnames(network) <- c("Source", "Target", "Score")
  length(network$Score)
  # network %>% slice_head(n = 50000)
  
  # Sort by Score in descending order and keep top 50,000 edges
  network %>%
    arrange(desc(Score)) %>%    # Sort by Score in descending order
      slice_sample(n = 50000)       # Keep top 50,000 edges
  #   sample(50000)       # Randomly select 50,000 edges
}

# Assuming you have a list of file paths for the networks
network_files <- list.files(path = network_dir, pattern = "*.csv", full.names = TRUE)
network_files

# Load and process each network
network_list <- map(network_files, read_and_sort_network)
# network_list[[1]]$Score

# Function to calculate Jaccard Index for two networks
calculate_jaccard <- function(network1, network2) {
  common_edges <- intersect(paste(network1$Source, network1$Target),
                            paste(network2$Source, network2$Target))
  union_edges <- union(paste(network1$Source, network1$Target),
                       paste(network2$Source, network2$Target))
  jaccard_index <- length(common_edges) / length(union_edges)
  return(jaccard_index)
}

# Initialize a matrix to store pairwise JI values
n <- length(network_list)
ji_matrix <- matrix(NA, n, n, dimnames = list(basename(network_files), basename(network_files)))

# Calculate Jaccard Index for each pair of networks
for (i in 1:(n - 1)) {
  for (j in (i + 1):n) {
    ji_matrix[i, j] <- calculate_jaccard(network_list[[i]], network_list[[j]])
    ji_matrix[j, i] <- ji_matrix[i, j]  # Mirror the value for symmetric matrix
  }
}

# View the Jaccard Index matrix
print(ji_matrix)
write.csv(ji_matrix,output_path)
# Extract unique pairwise JI values (exclude diagonal and duplicates)
ji_values <- ji_matrix[lower.tri(ji_matrix)]

# Calculate the median Jaccard Index
median_ji <- median(ji_values, na.rm = TRUE)

# Print the median JI
print(median_ji)
