# Aetheris Infrastructure
provider "aetheris_cloud" {
  region = "global-low-latency"
}
resource "game_node" "primary" {
  capacity = "50000-ccu"
  auto_scaling = true
}
