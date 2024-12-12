library(dplyr)
library(readr)
library(purrr)
library(ggplot2)

# Parameters
sample_name <- "macrophage_stability_buffer4"
network_dir <- paste0("OUTPUT/LINGER/", sample_name, "/STANDARDIZED_INFERRED_NETWORKS")
output_path <- paste0("OUTPUT/LINGER/", sample_name, "/jaccard_index_stability_matrix_bootstrap.csv")
num_bootstraps <- 10   # Number of bootstrap iterations
sample_size <- 50000    # Number of edges to sample in each bootstrap

# Function to read and rename columns
read_network <- function(file) {
  network <- read_csv(file)
  colnames(network) <- c("Source", "Target", "Score")
  return(network)
}

# Read all networks
network_files <- list.files(path = network_dir, pattern = "*.csv", full.names = TRUE)
network_list <- map(network_files, read_network)

# Function to calculate Jaccard Index
calculate_jaccard <- function(sampled_edges, network) {
  network_edges <- network %>% select(Source, Target)
  
  # Find intersection and union
  common_edges <- intersect(paste(sampled_edges$Source, sampled_edges$Target),
                            paste(network_edges$Source, network_edges$Target))
  union_edges <- union(paste(sampled_edges$Source, sampled_edges$Target),
                       paste(network_edges$Source, network_edges$Target))
  jaccard_index <- length(common_edges) / length(union_edges)
  return(jaccard_index)
}

# Function to perform bootstrapping and calculate Jaccard Index for a single network
bootstrap_jaccard <- function(reference_network, target_network, num_bootstraps, sample_size) {
  jaccard_indices <- replicate(num_bootstraps, {
    sampled_edges <- reference_network %>% slice_sample(n = sample_size)
    calculate_jaccard(sampled_edges, target_network)
  })
  mean(jaccard_indices)  # Return mean Jaccard index across bootstraps
}

# Initialize matrix to store bootstrap Jaccard indices
n <- length(network_list)
ji_matrix <- matrix(NA, n, n, dimnames = list(basename(network_files), basename(network_files)))

# Compute pairwise Jaccard indices with bootstrapping
for (i in 1:(n - 1)) {
  for (j in (i + 1):n) {
    ji_matrix[i, j] <- bootstrap_jaccard(network_list[[i]], network_list[[j]], num_bootstraps, sample_size)
    ji_matrix[j, i] <- ji_matrix[i, j]  # Symmetric matrix
  }
}

# Write the Jaccard index matrix to a CSV file
write.csv(ji_matrix, output_path, row.names = TRUE)

# Extract unique pairwise Jaccard indices
ji_values <- ji_matrix[lower.tri(ji_matrix)]

# Calculate and print the median Jaccard index
median_ji <- median(ji_values, na.rm = TRUE)
print(median_ji)

# Visualize Jaccard indices as a histogram
ggplot(data.frame(Jaccard_Index = ji_values), aes(x = Jaccard_Index)) +
  geom_histogram(binwidth = 0.01, fill = "steelblue", color = "black") +
  labs(title = "Histogram of Jaccard Indices (Bootstrapped)",
       x = "Jaccard Index",
       y = "Frequency") +
  theme_minimal()

# Save the histogram to a file
histogram_file <- paste0(dirname(output_path), "/jaccard_index_histogram.png")
ggsave(histogram_file, plot = histogram, width = 8, height = 6, dpi = 300)

print(paste("Histogram saved to:", histogram_file))
