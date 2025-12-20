#ifndef NVTOP_ROCM_SMI_UTILS_H
#define NVTOP_ROCM_SMI_UTILS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "nvtop/extract_gpuinfo_common.h"

bool nvtop_rocm_smi_init(void);
void nvtop_rocm_smi_shutdown(void);
bool nvtop_rocm_smi_is_available(void);

bool nvtop_rocm_smi_find_device(const char *pdev, uint32_t *out_index);
bool nvtop_rocm_smi_device_name(uint32_t index, char *name, size_t name_len);

void nvtop_rocm_smi_refresh_dynamic(uint32_t index, struct gpuinfo_dynamic_info *dynamic_info);

#endif // NVTOP_ROCM_SMI_UTILS_H
