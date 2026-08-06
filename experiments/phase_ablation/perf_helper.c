/* perf_helper.c
 *
 * In-process hardware counter reads for exactly the region the MLIR kernel
 * brackets (a single @spgemm call), avoiding both the whole-process
 * contamination of wrapping the binary in `perf stat` and the repeated
 * cold-starts of launching a fresh process per iteration.
 *
 * Opens 8 events as one perf_event_open() group with the "slots" fixed
 * counter as group leader: slots, topdown-retiring, topdown-bad-spec,
 * topdown-fe-bound, topdown-be-bound, instructions, cycles, branch-misses.
 * Intel's TopDown L1 metrics (retiring/bad-spec/fe-bound/be-bound) are
 * derived from a shared PERF_METRICS MSR and are only valid when opened as
 * members of a group led by "slots" — opening them independently fails
 * outright. branch-misses is the raw hardware branch-misprediction count,
 * distinct from bad_speculation_pct — TMA's Bad Speculation also includes
 * machine clears unrelated to branch prediction, so it's not the same
 * quantity.
 *
 * The topdown-* and slots raw event encodings (type + config) are read from
 * sysfs at init time rather than hardcoded, so this works unmodified across
 * different Intel CPU generations and both hybrid client PMUs (cpu_core)
 * and uniform server PMUs (cpu).
 *
 * Exposed to the MLIR kernel as plain functions (no structs across the
 * MLIR/C boundary): pe_init, pe_reset_enable, pe_disable, pe_read, pe_close.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <asm/unistd.h>
#include <linux/perf_event.h>

#define PE_COUNT 8
/* Index order shared with the MLIR kernel and run.sh's tma_raw parser. */
enum { PE_SLOTS = 0, PE_RETIRING, PE_BAD_SPEC, PE_FE_BOUND, PE_BE_BOUND,
       PE_INSTRUCTIONS, PE_CYCLES, PE_BRANCH_MISSES };

static int g_fds[PE_COUNT] = { -1, -1, -1, -1, -1, -1, -1, -1 };

static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                             int cpu, int group_fd, unsigned long flags) {
  return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

/* Reads "event=0xNN,umask=0xMM[,...]" from a sysfs event file and returns
 * config = event | (umask << 8), the standard Intel raw-event encoding. */
static int read_sysfs_config(const char *pmu_dir, const char *event_name,
                              uint64_t *config_out) {
  char path[256];
  snprintf(path, sizeof(path), "%s/events/%s", pmu_dir, event_name);
  FILE *f = fopen(path, "r");
  if (!f) return -1;
  char buf[256] = {0};
  size_t n = fread(buf, 1, sizeof(buf) - 1, f);
  fclose(f);
  if (n == 0) return -1;

  unsigned long event = 0, umask = 0;
  char *p;
  if ((p = strstr(buf, "event=")) == NULL) return -1;
  event = strtoul(p + 6, NULL, 0);
  if ((p = strstr(buf, "umask=")) != NULL) umask = strtoul(p + 6, NULL, 0);

  *config_out = event | (umask << 8);
  return 0;
}

static int read_sysfs_type(const char *pmu_dir, uint32_t *type_out) {
  char path[256];
  snprintf(path, sizeof(path), "%s/type", pmu_dir);
  FILE *f = fopen(path, "r");
  if (!f) return -1;
  int ok = (fscanf(f, "%u", type_out) == 1);
  fclose(f);
  return ok ? 0 : -1;
}

static int open_grouped(uint32_t type, uint64_t config, int group_fd) {
  struct perf_event_attr pe;
  memset(&pe, 0, sizeof(pe));
  pe.type = type;
  pe.size = sizeof(pe);
  pe.config = config;
  pe.disabled = (group_fd == -1) ? 1 : 0;
  pe.exclude_kernel = 1;
  pe.exclude_hv = 1;
  long fd = perf_event_open(&pe, 0, -1, group_fd, 0);
  if (fd < 0) {
    fprintf(stderr, "perf_helper: perf_event_open failed for type=%u config=0x%llx: %s\n",
            type, (unsigned long long)config, strerror(errno));
  }
  return (int)fd;
}

/* Returns 0 on success, -1 on failure (fds left at -1; pe_read then
 * returns 0 for everything rather than crashing the benchmark run). */
int64_t pe_init(void) {
  const char *pmu_dir = "/sys/bus/event_source/devices/cpu_core";
  if (access(pmu_dir, F_OK) != 0) pmu_dir = "/sys/bus/event_source/devices/cpu";

  uint32_t type;
  uint64_t cfg_slots, cfg_retiring, cfg_bad_spec, cfg_fe_bound, cfg_be_bound;
  if (read_sysfs_type(pmu_dir, &type) != 0 ||
      read_sysfs_config(pmu_dir, "slots", &cfg_slots) != 0 ||
      read_sysfs_config(pmu_dir, "topdown-retiring", &cfg_retiring) != 0 ||
      read_sysfs_config(pmu_dir, "topdown-bad-spec", &cfg_bad_spec) != 0 ||
      read_sysfs_config(pmu_dir, "topdown-fe-bound", &cfg_fe_bound) != 0 ||
      read_sysfs_config(pmu_dir, "topdown-be-bound", &cfg_be_bound) != 0) {
    fprintf(stderr, "perf_helper: failed to resolve topdown event encodings under %s\n", pmu_dir);
    return -1;
  }

  g_fds[PE_SLOTS]    = open_grouped(type, cfg_slots, -1);
  g_fds[PE_RETIRING] = open_grouped(type, cfg_retiring, g_fds[PE_SLOTS]);
  g_fds[PE_BAD_SPEC] = open_grouped(type, cfg_bad_spec, g_fds[PE_SLOTS]);
  g_fds[PE_FE_BOUND] = open_grouped(type, cfg_fe_bound, g_fds[PE_SLOTS]);
  g_fds[PE_BE_BOUND] = open_grouped(type, cfg_be_bound, g_fds[PE_SLOTS]);
  g_fds[PE_INSTRUCTIONS] = open_grouped(PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, g_fds[PE_SLOTS]);
  g_fds[PE_CYCLES]       = open_grouped(PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, g_fds[PE_SLOTS]);
  g_fds[PE_BRANCH_MISSES] = open_grouped(PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES, g_fds[PE_SLOTS]);

  for (int i = 0; i < PE_COUNT; i++) {
    if (g_fds[i] < 0) return -1;
  }
  return 0;
}

void pe_reset_enable(void) {
  if (g_fds[PE_SLOTS] < 0) return;
  ioctl(g_fds[PE_SLOTS], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
  ioctl(g_fds[PE_SLOTS], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
}

void pe_disable(void) {
  if (g_fds[PE_SLOTS] < 0) return;
  ioctl(g_fds[PE_SLOTS], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
}

/* idx: 0=slots 1=retiring 2=bad_spec 3=fe_bound 4=be_bound 5=instructions
 * 6=cycles 7=branch_misses */
int64_t pe_read(int32_t idx) {
  if (idx < 0 || idx >= PE_COUNT || g_fds[idx] < 0) return 0;
  long long v = 0;
  if (read(g_fds[idx], &v, sizeof(v)) != (ssize_t)sizeof(v)) return 0;
  return v;
}

void pe_close(void) {
  for (int i = 0; i < PE_COUNT; i++) {
    if (g_fds[i] >= 0) { close(g_fds[i]); g_fds[i] = -1; }
  }
}
