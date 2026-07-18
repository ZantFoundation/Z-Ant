# Z-Ant Model Performance Comparison

## ReleaseFast Optimization

| Model | Total Instructions | Exec Time (us) | Primary Alloc | Alloc % | Total Heap Usage | Memory Leaks | Executable Size | Peak Memory | Allocations | time dyn us | time static us |
|-------|-------------------|----------------|---------------|---------|------------------|--------------|-----------------|-------------|-------------|-------------|-------------|
| beer | 87,791,919 | 605 | 36,960B | 99.91% | 2,494,460B | 1,728B | 1.7M | 36.96 KB | 163 | fail | fail |
| new2 | 2,376,299,678 | 2,420 | 102,208B | 99.96% | 1,554,336B | 320B | 9.5M | 102.2 KB | 33,353 | 11,363,927 | 10,286,534 |
| darknet_s | 72,170,395 | 697 | 45,120B | 99.93% | 650,352B | 16B | 19M | 45.12 KB | 259 | NA | NA |
| fomo8 | 5,801,221 | 434 | 110,652B | 99.95% | 286,025B | 24B | 1.2M | 110.7 KB | 17 | fail | fail |
| mnist-8 | 15,218,762 | 5014 | 50,240B | 99.94% | 113,744B | 40B | 1.4M | 50.24 KB | 1,877 | 189,293 | 169,364 |
| mobilenet_v2 | 75,913,090 | 648 | 45,120B | 99.93% | 592,080B | 16B | 8.5M | 45.12 KB | 135 |  1,445,338 | 1,211,108 |
| coco80_q | 1,902,010,879 | 4,187 | 112,192B | 99.96% | 1,554,336B | 320B | 9.6M | 112.2 KB | 33,353 | 12,049,579 | tbd |


## ReleaseSmall Optimization

| Model | Total Instructions | Exec Time (us) | Primary Alloc | Alloc % | Total Heap Usage | Memory Leaks | Executable Size | Peak Memory | Allocations | time dyn us | time static us |
|-------|-------------------|----------------|---------------|---------|------------------|--------------|-----------------|-------------|-------------|-------------|-------------|
| beer | 130,930,476 | 558 | 73,824B | 99.95% | 2,494,492B | 1,728B | 88K | 73.82 KB | 164 | fail | fail |
| new2 | 5,638,188,336 | 4,920 | 112,192B | 99.96% | 6,199,296B | 320B | 9.1M | 112.2 KB | 323,660 | 1,563,288 | 1,400,959 |
| darknet_s | 106,000,990 | 608 | 45,120B | 99.93% | 650,416B | 16B | 17M | 45.12 KB | 267 | NA | NA |
| fomo8 | 5,724,155 | 428 | 110,652B | 99.95% | 286,025B | 24B | 182K | 110.7 KB | 17 | fail | fail |
| mnist-8 | 14,401,577 | 9238 | 50,240B | 99.94% | 113,744B | 40B | 50K | 50.24 KB | 1,877 | 143,507 | 128,454 |
| mobilenet_v2 | 105,265,510 | 552 | 45,120B | 99.93% | 592,080B | 16B | 7.0M | 45.12 KB | 135 | 1,614,847 | 1,445,338 |
| coco80_q | 2,062,059,985 | 3,879 | 112,192B | 99.96% | 6,199,296B | 320B | 9.1M | 112.2 KB | 323,660 | tbd | tbd |


## Comparative Analysis (ReleaseFast vs ReleaseSmall)

| Model | Instructions Δ | Exec Time Δ | Size Reduction | Peak Memory Δ | Allocations Δ | time dyn us | time static us |
|-------|----------------|-------------|----------------|---------------|---------------|------------|------------|
| beer | +49.1% | -7.8% | 95% (1.7M→88K) | +99.8% | +0.6% | fail→fail | fail→fail |
| new2 | +137.3% | +103.3% | 4% (9.5M→9.1M) | +9.8% | +870.0% | -86.2% (11.4s→1.6s) | -86.4% (10.3s→1.4s) |
| darknet_s | +46.9% | -12.8% | 11% (19M→17M) | 0% | +3.1% | NA | NA |
| fomo8 | -1.3% | -1.4% | 85% (1.2M→182K) | 0% | 0% | fail→fail | fail→fail |
| mnist-8 | -5.4% | +6.2% | 96% (1.4M→50K) | 0% | 0% | -24.2% (189ms→144ms) | -24.1% (169ms→128ms) |
| mobilenet_v2 | +38.7% | -14.8% | 18% (8.5M→7.0M) | 0% | 0% | +1025% (144ms→1.6s) | +1025% (128ms→1.4s) |
| coco80_q | +8.4% | -7.4% | 5% (9.6M→9.1M) | 0% | +870.0% | tbd | tbd |


# Zant (time us, static allocation)
|  Model  |   native fast   |   native small   |   cortex-m7 fast   |   cortex-m7 small   |
|---------|-----------------|------------------|--------------------|-------------------- |
| beer    | | | | |
| mnist-8 | 5014            | 9238             | 128,454            | 169,364             |
| mobilenet_v2| 8363        | 12163            | 1,445,338          | 1,445,338           |

# TFL (time us)
| Model       |   TFL    |   EI    |
|-------------|----------|---------|
| beer        | | |
| mnist-8     |         | 197000 us | 
| mobilenet_v2|         | 304000 us, 282.8K RAM, 6.9M Flash |

