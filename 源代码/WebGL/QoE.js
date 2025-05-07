export class QoEManager {
    constructor(representations) {
      this.resetState(representations);
    }
  
    resetState(representations) {
      this.state = {
        totalQuality: 0,
        totalStalling: 0,
        totalTemporalSmoothness: 0,
        totalSpatialSmoothness: 0,
        lastQuality: null,
        lastChunkQuality: {},
        maxBitrate: Math.max(...representations.map(r => r[0].bandwidth)) / 1000000, 
        chunkCount: 0,  
        startTime: performance.now(),
        lastChunkSize: 5 
      };
    }
  
    calculateChunkQoE(chunkNumber, qualityLevel, chunkBitrate, isRebuffering = false) {

      const bitrateMbps = chunkBitrate / 1000000; 
      const avgFrameBitrate = bitrateMbps / this.state.lastChunkSize;
      
 
      const visibleQuality = bitrateMbps; 
      this.state.totalQuality += visibleQuality;
      
      if (isRebuffering) {
        const stallingPenalty = 2 * this.state.maxBitrate;
        this.state.totalStalling += stallingPenalty;
      }
      
      if (this.state.lastQuality !== null) {
        const qualityDiff = Math.abs(avgFrameBitrate - (this.state.lastQuality / this.state.lastChunkSize));
        this.state.totalTemporalSmoothness += qualityDiff * 0.5;
      }
      

      if (this.state.lastChunkQuality[chunkNumber] !== undefined) {
        const lastAvgBitrate = this.state.lastChunkQuality[chunkNumber] / this.state.lastChunkSize;
        const spatialDiff = Math.abs(avgFrameBitrate - lastAvgBitrate);
        this.state.totalSpatialSmoothness += spatialDiff * 0.5;
      }
      
      this.state.lastQuality = bitrateMbps;
      this.state.lastChunkQuality[chunkNumber] = bitrateMbps;
      this.state.chunkCount++;
      
      return this.getCurrentMetrics();
    }
  
    getCurrentMetrics() {
      return {
        visibleQuality: this.state.totalQuality / this.state.chunkCount || 0,
        stallingPenalty: this.state.totalStalling / this.state.chunkCount || 0,
        temporalSmoothness: this.state.totalTemporalSmoothness / this.state.chunkCount || 0,
        spatialSmoothness: this.state.totalSpatialSmoothness / this.state.chunkCount || 0,
        averageQoE: (this.state.totalQuality - this.state.totalStalling - 
                    this.state.totalTemporalSmoothness - this.state.totalSpatialSmoothness) / 
                    this.state.chunkCount || 0,
        bitrate: this.state.lastQuality,
        duration: (performance.now() - this.state.startTime) / 1000,
        totalChunks: this.state.chunkCount
      };
    }
    setCurrentChunkSize(frameCount) {
      this.state.lastChunkSize = frameCount;
    }
  }