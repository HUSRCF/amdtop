#include "nvtop/rocm_smi_utils.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
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

  rsmi_status_t status = rsmi_init(0);
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

  // 使用更大的临时缓冲区防止溢出
  char temp_name[256];
  memset(temp_name, 0, sizeof(temp_name));

  rsmi_status_t status = rsmi_dev_name_get(index, temp_name, sizeof(temp_name) - 1);

  if (status == RSMI_STATUS_SUCCESS && temp_name[0] != '\0') {
    strncpy(name, temp_name, name_len - 1);
    name[name_len - 1] = '\0';
    return true;
  }

  name[0] = '\0';
  return false;
}

static bool rsmi_get_clock_mhz(uint32_t index, rsmi_clk_type_t type, unsigned int *current, unsigned int *max) {
  // 暂时禁用时钟频率读取，因为 rsmi_dev_gpu_clk_freq_get 可能导致栈溢出
  // 这可能是由于 ROCm SMI 库版本与头文件不匹配导致的
  (void)index;
  (void)type;
  (void)current;
  (void)max;
  return false;
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

  // Get power draw (try different APIs for compatibility)
  uint64_t power = 0;
  if (rsmi_dev_power_ave_get(index, 0, &power) == RSMI_STATUS_SUCCESS) {
    SET_GPUINFO_DYNAMIC(dynamic_info, power_draw, (unsigned int)(power / 1000));
  }

  uint64_t cap = 0;
  if (rsmi_dev_power_cap_get(index, 0, &cap) != RSMI_STATUS_SUCCESS || cap == 0) {
    cap = 0;
    (void)rsmi_dev_power_cap_default_get(index, &cap);
  }
  if (cap > 0) {
    SET_GPUINFO_DYNAMIC(dynamic_info, power_draw_max, (unsigned int)(cap / 1000));
  }

  // Note: GPU metrics API may not be available in all ROCm versions
  // Commented out to maintain compatibility
  /*
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
        // Note: pcie_bandwidth_inst may not be available in all ROCm versions
        // Commented out to maintain compatibility
      }
    }
  }
  */

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

bool nvtop_rocm_smi_get_processes(struct gpu_info *gpu_info) {
  if (!rsmi_ready || !gpu_info)
    return false;

  // 第一次调用：获取进程数量
  uint32_t num_items = 0;
  rsmi_status_t status = rsmi_compute_process_info_get(NULL, &num_items);

  if (status != RSMI_STATUS_SUCCESS || num_items == 0) {
    // 没有进程或调用失败
    gpu_info->processes_count = 0;
    return true;
  }

  // 分配临时内存存储 ROCm SMI 进程信息
  rsmi_process_info_t *procs = calloc(num_items, sizeof(rsmi_process_info_t));
  if (!procs)
    return false;

  // 第二次调用：获取实际进程数据
  status = rsmi_compute_process_info_get(procs, &num_items);
  if (status != RSMI_STATUS_SUCCESS) {
    free(procs);
    return false;
  }

  // 确保 processes 数组有足够空间
  if (num_items > gpu_info->processes_array_size) {
    struct gpu_process *new_processes = realloc(gpu_info->processes, num_items * sizeof(struct gpu_process));
    if (!new_processes) {
      free(procs);
      return false;
    }
    gpu_info->processes = new_processes;
    gpu_info->processes_array_size = num_items;
  }

  // 清空现有进程信息
  memset(gpu_info->processes, 0, num_items * sizeof(struct gpu_process));

  // 转换 ROCm SMI 进程信息到 AMDTOP 格式
  gpu_info->processes_count = num_items;
  for (uint32_t i = 0; i < num_items; ++i) {
    struct gpu_process *proc = &gpu_info->processes[i];

    proc->pid = procs[i].process_id;
    proc->type = gpu_process_compute; // ROCm SMI 只报告计算进程

    // 设置显存使用量（ROCm SMI 返回字节）
    if (procs[i].vram_usage > 0) {
      SET_GPUINFO_PROCESS(proc, gpu_memory_usage, procs[i].vram_usage);
    }

    // 设置 GPU 使用率（从 CU 占用率转换）
    if (procs[i].cu_occupancy != 0xFFFFFFFF) { // CU_OCCUPANCY_INVALID
      SET_GPUINFO_PROCESS(proc, gpu_usage, procs[i].cu_occupancy);
    }
  }

  free(procs);
  return true;
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

bool nvtop_rocm_smi_get_processes(struct gpu_info *gpu_info) {
  (void)gpu_info;
  return false;
}

#endif
