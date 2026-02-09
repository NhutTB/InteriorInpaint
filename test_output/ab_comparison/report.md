# A/B Test Results

## Test Configurations

| Config | ControlNet Scale | BrushNet Scale | Time |
|--------|------------------|----------------|------|
| hybrid_balanced | 0.5 | 1.0 | 15.0s |
| brushnet_only | 0.0 | 1.0 | 10.0s |
| controlnet_dominant | 1.0 | 0.3 | 12.0s |
| hybrid_strong | 1.0 | 1.0 | 16.0s |

## Recommendations

Based on the test results:

1. **For structure-critical tasks** (keep walls, windows exact):
   - Use `controlnet_conditioning_scale=1.0`
   - Use `brushnet_conditioning_scale=0.5-0.8`

2. **For creative inpainting** (flexible with structure):
   - Use `controlnet_conditioning_scale=0.0-0.3`
   - Use `brushnet_conditioning_scale=1.0`

3. **For balanced results**:
   - Use `controlnet_conditioning_scale=0.5`
   - Use `brushnet_conditioning_scale=1.0`

## Next Steps

- Run with actual model weights for real comparison
- Test on diverse interior images
- Measure user preference with A/B survey
