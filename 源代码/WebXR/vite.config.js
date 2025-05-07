import { defineConfig } from 'vite';
import basicSsl from '@vitejs/plugin-basic-ssl'
export default {
  plugins: [basicSsl()],
  server: {
    host: '0.0.0.0', // 监听所有接口
    port: 5173,
    https:true
  },
};
