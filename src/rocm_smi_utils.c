#include "nvtop/rocm_smi_utils.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#ifdef HAVE_ROCM_SMI
#include <rocm_smi/rocm_smi.h>

static bool rsmi_ready = false;
static uint32_t rsmi_device_count = 0;

static void rsmi_bdf_to_pdev(uint64_t bdfid, char *buf, size_t len) {
  uint32_t domain = (uint32_t)(bdfid >> 32);
  uint32_t bus = (bdfid >> 8) & 0xffU;
  uint32_t dev = (bdfid >> 3) & 0x1fU;
  uint32_t func = bdfid & 0x7U;
  snprintf(buf, len, "%04x:%02x:%02x.%u", domain, bus, dev, func);
}

bool nvtop_rocm_smi_init(void) {
  if (rsmi_ready)
    return true;

  rsmi_status_t status = rsmi_init(RSMI_INIT_FLAG_THRAD_ONLY_MUTEX);
  if (status != RSMI_STATUS_SUCCESS)
    return false;

  status = rsmi_num_monitor_devices(&rsmi_device_count);
  if (status != RSMI_STATUS_SUCCESS) {
    rsmi_shut_down();
    rsmi_device_count = 0;
    return false;
  }

  rsmi_ready = true;
  return true;
}

void nvtop_rocm_smi_shutdown(void) {
  if (!rsmi_ready)
    return;

  rsmi_shut_down();
  rsmi_device_count = 0;
  rsmi_ready = false;
}

bool nvtop_rocm_smi_is_available(void) { return rsmi_ready; }

bool nvtop_rocm_smi_find_device(const char *pdev, uint32_t *out_index) {
  if (!rsmi_ready || !pdev || !out_index)
    return false;

  for (uint32_t i = 0; i < rsmi_device_count; ++i) {
    uint64_t bdfid = 0;
    if (rsmi_dev_pci_id_get(i, &bdfid) != RSMI_STATUS_SUCCESS)
      continue;

    char bdf_str[32];
    rsmi_bdf_to_pdev(bdfid, bdf_str, sizeof(bdf_str));
    if (strcmp(bdf_str, pdev) == 0) {
      *out_index = i;
      return true;
    }
  }

  return false;
}

bool nvtop_rocm_smi_device_name(uint32_t index, char *name, size_t name_len) {
  if (!rsmi_ready || !name || name_len == 0)
    return false;

  name[0] = '\0';
  rsmi_status_t status = rsmi_dev_market_name_get(index, name, name_len);
  if (status != RSMI_STATUS_SUCCESS || name[0] == '\0') {
    status = rsmi_dev_name_get(index, name, name_len);
  }

  return (status == RSMI_STATUS_SUCCESS && name[0] != '\0');
}

static bool rsmi_get_clock_mhz(uint32_t index, rsmi_clk_type_t type, unsigned int *current, unsigned int *max) {
  rsmi_frequencies_t freqs;
  rsmi_status_t status = rsmi_dev_gpu_clk_freq_get(index, type, &freqs);
  if (status != RSMI_STATUS_SUCCESS || freqs.num_supported == 0)
    return false;

  uint64_t max_freq = 0;
  for (uint32_t i = 0; i < freqs.num_supported; ++i) {
    if (freqs.frequency[i] > max_freq)
      max_freq = freqs.frequency[i];
  }

  if (max && max_freq > 0)
    *max = (unsigned int)(max_freq / 1000000ULL);

  if (current && freqs.current < freqs.num_supported) {
    uint64_t cur = freqs.frequency[freqs.current];
    if (cur > 0)
      *current = (unsigned int)(cur / 1000000ULL);
  }

  return true;
}

void nvtop_rocm_smi_refresh_dynamic(uint32_t index, struct gpuinfo_dynamic_info *dynamic_info) {
  if (!rsmi_ready || !dynamic_info)
    return;

  uint32_t busy = 0;
  if (rsmi_dev_busy_percent_get(index, &busy) == RSMI_STATUS_SUCCESS) {
    SET_GPUINFO_DYNAMIC(dynamic_info, gpu_util_rate, busy);
  }

  unsigned int cur = 0;
  unsigned int max = 0;
  if (rsmi_get_clock_mhz(index, RSMI_CLK_TYPE_SYS, &cur, &max)) {
    if (cur > 0)
      SET_GPUINFO_DYNAMIC(dynamic_info, gpu_clock_speed, cur);
    if (max > 0)
      SET_GPUINFO_DYNAMIC(dynamic_info, gpu_clock_speed_max, max);
  }

  cur = 0;
  max = 0;
  if (rsmi_get_clock_mhz(index, RSMI_CLK_TYPE_MEM, &cur, &max)) {
    if (cur > 0)
      SET_GPUINFO_DYNAMIC(dynamic_info, mem_clock_speed, cur);
    if (max > 0)
      SET_GPUINFO_DYNAMIC(dynamic_info, mem_clock_speed_max, max);
  }

  uint64_t total = 0;
  uint64_t used = 0;
  if (rsmi_dev_memory_total_get(index, RSMI_MEM_TYPE_VRAM, &total) == RSMI_STATUS_SUCCESS) {
    SET_GPUINFO_DYNAMIC(dynamic_info, total_memory, total);
  }
  if (rsmi_dev_memory_usage_get(index, RSMI_MEM_TYPE_VRAM, &used) == RSMI_STATUS_SUCCESS) {
    SET_GPUINFO_DYNAMIC(dynamic_info, used_memory, used);
  }

  if (GPUINFO_DYNAMIC_FIELD_VALID(dynamic_info, total_memory) &&
      GPUINFO_DYNAMIC_FIELD_VALID(dynamic_info, used_memory) && dynamic_info->total_memory > 0) {
    unsigned long long free_mem = dynamic_info->total_memory - dynamic_info->used_memory;
    SET_GPUINFO_DYNAMIC(dynamic_info, free_memory, free_mem);
    SET_GPUINFO_DYNAMIC(dynamic_info, mem_util_rate,
                        (dynamic_info->used_memory * 100) / dynamic_info->total_memory);
  } else {
    uint32_t mem_busy = 0;
    if (rsmi_dev_memory_busy_percent_get(index, &mem_busy) == RSMI_STATUS_SUCCESS) {
      SET_GPUINFO_DYNAMIC(dynamic_info, mem_util_rate, mem_busy);
    }
  }

  int64_t temp = 0;
  if (rsmi_dev_temp_metric_get(index, RSMI_TEMP_TYPE_EDGE, RSMI_TEMP_CURRENT, &temp) == RSMI_STATUS_SUCCESS) {
    SET_GPUINFO_DYNAMIC(dynamic_info, gpu_temp, (unsigned int)(temp / 1000));
  }

  int64_t temp_junc = 0;
  if (rsmi_dev_temp_metric_get(index, RSMI_TEMP_TYPE_JUNCTION, RSMI_TEMP_CURRENT, &temp_junc) ==
      RSMI_STATUS_SUCCESS) {
    SET_GPUINFO_DYNAMIC(dynamic_info, gpu_temp_junction, (unsigned int)(temp_junc / 1000));
  }

  int64_t temp_mem = 0;
  if (rsmi_dev_temp_metric_get(index, RSMI_TEMP_TYPE_MEMORY, RSMI_TEMP_CURRENT, &temp_mem) ==
      RSMI_STATUS_SUCCESS) {
    SET_GPUINFO_DYNAMIC(dynamic_info, gpu_temp_mem, (unsigned int)(temp_mem / 1000));
  }

  int64_t fan_speed = -1;
  if (rsmi_dev_fan_speed_get(index, 0, &fan_speed) == RSMI_STATUS_SUCCESS && fan_speed >= 0) {
    SET_GPUINFO_DYNAMIC(dynamic_info, fan_speed, (unsigned int)(fan_speed * 100 / RSMI_MAX_FAN_SPEED));
  }

  int64_t fan_rpm = -1;
  if (rsmi_dev_fan_rpms_get(index, 0, &fan_rpm) == RSMI_STATUS_SUCCESS && fan_rpm >= 0) {
    SET_GPUINFO_DYNAMIC(dynamic_info, fan_rpm, (unsigned int)fan_rpm);
  }

  uint64_t power = 0;
  RSMI_POWER_TYPE power_type = RSMI_INVALID_POWER;
  if (rsmi_dev_power_get(index, &power, &power_type) == RSMI_STATUS_SUCCESS) {
    SET_GPUINFO_DYNAMIC(dynamic_info, power_draw, (unsigned int)(power / 1000));
  } else if (rsmi_dev_power_ave_get(index, 0, &power) == RSMI_STATUS_SUCCESS) {
    SET_GPUINFO_DYNAMIC(dynamic_info, power_draw, (unsigned int)(power / 1000));
  }

  uint64_t cap = 0;
  if (rsmi_dev_power_cap_get(index, 0, &cap) == RSMI_STATUS_SUCCESS) {
    SET_GPUINFO_DYNAMIC(dynamic_info, power_draw_max, (unsigned int)(cap / 1000));
  }

  metrics_table_header_t header;
  if (rsmi_dev_metrics_header_info_get(index, &header) == RSMI_STATUS_SUCCESS) {
    rsmi_gpu_metrics_t metrics;
    if (rsmi_dev_gpu_metrics_info_get(index, &metrics) == RSMI_STATUS_SUCCESS) {
      if (metrics.pcie_link_width > 0 && metrics.pcie_link_width != UINT16_MAX) {
        SET_GPUINFO_DYNAMIC(dynamic_info, pcie_link_width, metrics.pcie_link_width);
      }
      if (metrics.pcie_link_speed > 0 && metrics.pcie_link_speed != UINT16_MAX) {
        unsigned speed_gt = (metrics.pcie_link_speed + 5) / 10;
        unsigned gen = nvtop_pcie_gen_from_link_speed(speed_gt);
        if (gen > 0)
          SET_GPUINFO_DYNAMIC(dynamic_info, pcie_link_gen, gen);
      }

      if (!GPUINFO_DYNAMIC_FIELD_VALID(dynamic_info, pcie_rx) ||
          !GPUINFO_DYNAMIC_FIELD_VALID(dynamic_info, pcie_tx)) {
        if (metrics.pcie_bandwidth_inst != UINT64_MAX) {
          double total_kib = ((double)metrics.pcie_bandwidth_inst / 8.0) * 1024.0;
          unsigned int half = (unsigned int)llround(total_kib / 2.0);
          unsigned int rest = (unsigned int)llround(total_kib) - half;
          SET_GPUINFO_DYNAMIC(dynamic_info, pcie_rx, half);
          SET_GPUINFO_DYNAMIC(dynamic_info, pcie_tx, rest);
        }
      }
    }
  }

  uint64_t sent = 0, received = 0, max_pkt = 0;
  if (rsmi_dev_pci_throughput_get(index, &sent, &received, &max_pkt) == RSMI_STATUS_SUCCESS) {
    uint64_t sent_bytes = sent;
    uint64_t received_bytes = received;
    if (max_pkt > 0) {
      sent_bytes *= max_pkt;
      received_bytes *= max_pkt;
    }
    SET_GPUINFO_DYNAMIC(dynamic_info, pcie_tx, (unsigned int)(sent_bytes / 1024));
    SET_GPUINFO_DYNAMIC(dynamic_info, pcie_rx, (unsigned int)(received_bytes / 1024));
  }
}

#else

bool nvtop_rocm_smi_init(void) { return false; }
void nvtop_rocm_smi_shutdown(void) {}
bool nvtop_rocm_smi_is_available(void) { return false; }

bool nvtop_rocm_smi_find_device(const char *pdev, uint32_t *out_index) {
  (void)pdev;
  (void)out_index;
  return false;
}

bool nvtop_rocm_smi_device_name(uint32_t index, char *name, size_t name_len) {
  (void)index;
  if (name && name_len > 0)
    name[0] = '\0';
  return false;
}

void nvtop_rocm_smi_refresh_dynamic(uint32_t index, struct gpuinfo_dynamic_info *dynamic_info) {
  (void)index;
  (void)dynamic_info;
}

#endif
