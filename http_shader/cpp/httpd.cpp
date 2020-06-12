// This C++ shader is autogenerated by spirv-cross.
#include "spirv_cross/internal_interface.hpp"
#include "spirv_cross/external_interface.h"
#include <array>
#include <stdint.h>

using namespace spirv_cross;
using namespace glm;

namespace Impl
{
    struct Shader
    {
        struct Resources : ComputeResources
        {
            struct inputBuffer
            {
                ivec4 inputBytes[1];
            };
            
            internal::Resource<inputBuffer> _113__;
#define _113 __res->_113__.get()
            
            struct heapBuffer
            {
                ivec4 heap[1];
            };
            
            internal::Resource<heapBuffer> _216__;
#define _216 __res->_216__.get()
            
            struct outputBuffer
            {
                ivec4 outputBytes[1];
            };
            
            internal::Resource<outputBuffer> _236__;
#define _236 __res->_236__.get()
            
            inline void init(spirv_cross_shader& s)
            {
                ComputeResources::init(s);
                s.register_resource(_113__, 0, 0);
                s.register_resource(_216__, 0, 2);
                s.register_resource(_236__, 0, 1);
            }
        };
        
        Resources* __res;
        ComputePrivateResources __priv_res;
        
        inline int32_t getE(const ivec4 &v, const int32_t &i)
        {
            int32_t value = v.x;
            if (i == 1)
            {
                value = v.y;
            }
            else
            {
                if (i == 2)
                {
                    value = v.z;
                }
                else
                {
                    if (i == 3)
                    {
                        value = v.w;
                    }
                }
            }
            return value;
        }
        
        inline void setE(ivec4 &v, const int32_t &i, const int32_t &value)
        {
            if (i == 0)
            {
                v.x = value;
            }
            else
            {
                if (i == 1)
                {
                    v.y = value;
                }
                else
                {
                    if (i == 2)
                    {
                        v.z = value;
                    }
                    else
                    {
                        v.w = value;
                    }
                }
            }
        }
        
        inline void main()
        {
            int32_t wgId = int32_t(gl_GlobalInvocationID.x) * 4096;
            for (int32_t j = 0; j < 4096; j++)
            {
                int32_t reqOff = (wgId + j) * 64;
                int32_t resOff = (wgId + j) * 64;
                ivec4 requestInfo = _113.inputBytes[reqOff];
                if (requestInfo.x == 0)
                {
                    continue;
                }
                ivec4 req = _113.inputBytes[reqOff + 1];
                ivec4 req2 = _113.inputBytes[reqOff + 2];
                int32_t method = req.x;
                int32_t i = resOff;
                if (method == 542393671)
                {
                    int32_t key = ((((((((((req.y >> 8) & 255) - 48) * 1000000) + ((((req.y >> 16) & 255) - 48) * 100000)) + ((((req.y >> 24) & 255) - 48) * 10000)) + ((((req.z >> 0) & 255) - 48) * 1000)) + ((((req.z >> 8) & 255) - 48) * 100)) + ((((req.z >> 16) & 255) - 48) * 10)) + ((((req.z >> 24) & 255) - 48) * 1)) * 64;
                    int32_t _205 = key;
                    int32_t _207 = key;
                    bool _210 = (_205 >= 0) && (_207 < 33554432);
                    bool _222;
                    if (_210)
                    {
                        _222 = _216.heap[key].x > 0;
                    }
                    else
                    {
                        _222 = _210;
                    }
                    bool _230;
                    if (_222)
                    {
                        _230 = _216.heap[key].x <= 976;
                    }
                    else
                    {
                        _230 = _222;
                    }
                    if (_230)
                    {
                        _236.outputBytes[i + 0] = ivec4(32 + _216.heap[key].x, 0, 0, 0);
                        _236.outputBytes[i + 1] = ivec4(540028978, 1210075983, 793793620, 221326897);
                        _236.outputBytes[i + 2] = ivec4(1852793610, 1953391988, 1887007789, 538983013);
                        int32_t len = (_216.heap[key].x / 16) + int32_t((_216.heap[key].x % 16) > 0);
                        for (int32_t k = 0; k < len; k++)
                        {
                            _236.outputBytes[(i + 3) + k] = _216.heap[(key + 1) + k];
                        }
                        continue;
                    }
                }
                else
                {
                    if (method == 1414745936)
                    {
                        int32_t key_1 = ((((((((((req.y >> 16) & 255) - 48) * 1000000) + ((((req.y >> 24) & 255) - 48) * 100000)) + ((((req.z >> 0) & 255) - 48) * 10000)) + ((((req.z >> 8) & 255) - 48) * 1000)) + ((((req.z >> 16) & 255) - 48) * 100)) + ((((req.z >> 24) & 255) - 48) * 10)) + ((((req.w >> 0) & 255) - 48) * 1)) * 64;
                        if ((key_1 >= 0) && (key_1 < 33554432))
                        {
                            int32_t rnrn = 0;
                            int32_t readStart = 0;
                            int32_t readEnd = 512;
                            ivec4 w = ivec4(0);
                            int32_t l = 0;
                            int32_t hi = 0;
                            for (int32_t k_1 = 13; (k_1 < 1024) && (k_1 < 1024); k_1++)
                            {
                                int32_t v4i = k_1 / 16;
                                int32_t vi = k_1 - (v4i * 16);
                                int32_t c = vi / 4;
                                int32_t b = vi - (c * 4);
                                ivec4 param = _113.inputBytes[(reqOff + 1) + v4i];
                                int32_t param_1 = c;
                                int32_t chr = (getE(param, param_1) >> (b * 8)) & 255;
                                if (readStart > 0)
                                {
                                    if (chr == 0)
                                    {
                                        readEnd = k_1;
                                        break;
                                    }
                                    int32_t wc = l / 4;
                                    int32_t wb = l - (wc * 4);
                                    ivec4 param_2 = w;
                                    int32_t param_3 = wc;
                                    ivec4 param_4 = w;
                                    int32_t param_5 = wc;
                                    int32_t param_6 = getE(param_2, param_3) | (chr << (wb * 8));
                                    setE(param_4, param_5, param_6);
                                    w = param_4;
                                    l++;
                                    if (l == 16)
                                    {
                                        _216.heap[(key_1 + 1) + hi] = w;
                                        hi++;
                                        w *= ivec4(0);
                                        l = 0;
                                    }
                                }
                                else
                                {
                                    int32_t _466 = chr;
                                    bool _467 = _466 == 13;
                                    bool _473;
                                    if (_467)
                                    {
                                        _473 = (rnrn & 1) == 0;
                                    }
                                    else
                                    {
                                        _473 = _467;
                                    }
                                    if (_473)
                                    {
                                        rnrn++;
                                    }
                                    else
                                    {
                                        int32_t _479 = chr;
                                        bool _480 = _479 == 10;
                                        bool _486;
                                        if (_480)
                                        {
                                            _486 = (rnrn & 1) == 1;
                                        }
                                        else
                                        {
                                            _486 = _480;
                                        }
                                        if (_486)
                                        {
                                            rnrn++;
                                            if (rnrn == 4)
                                            {
                                                readStart = k_1;
                                            }
                                        }
                                        else
                                        {
                                            rnrn = 0;
                                        }
                                    }
                                }
                            }
                            int32_t _499 = l;
                            bool _500 = _499 > 0;
                            bool _506;
                            if (_500)
                            {
                                _506 = (1 + hi) < 64;
                            }
                            else
                            {
                                _506 = _500;
                            }
                            if (_506)
                            {
                                _216.heap[(key_1 + 1) + hi] = w;
                            }
                            _216.heap[key_1].x = readEnd - readStart;
                            _236.outputBytes[i + 0] = ivec4(48, 0, 0, 0);
                            _236.outputBytes[i + 1] = ivec4(540028978, 1210075983, 793793620, 221326897);
                            _236.outputBytes[i + 2] = ivec4(1852793610, 1953391988, 1887007789, 1948269157);
                            _236.outputBytes[i + 3] = ivec4(796162149, 1767992432, 218762606, 776687370);
                            continue;
                        }
                    }
                }
                _236.outputBytes[i + 0] = ivec4(34, 0, 0, 0);
                _236.outputBytes[i + 1] = ivec4(540028981, 541344066, 1347703880, 825110831);
                _236.outputBytes[i + 2] = ivec4(168626701, req.x, req.y, req.z);
                _236.outputBytes[i + 3] = ivec4(req.w, req2.x, req2.y, req2.z);
            }
        }
        
    };
}

spirv_cross_shader_t *spirv_cross_construct(void)
{
    return new ComputeShader<Impl::Shader, Impl::Shader::Resources, 4, 1, 1>();
}

void spirv_cross_destruct(spirv_cross_shader_t *shader)
{
    delete static_cast<ComputeShader<Impl::Shader, Impl::Shader::Resources, 4, 1, 1>*>(shader);
}

void spirv_cross_invoke(spirv_cross_shader_t *shader)
{
    static_cast<ComputeShader<Impl::Shader, Impl::Shader::Resources, 4, 1, 1>*>(shader)->invoke();
}

static const struct spirv_cross_interface vtable =
{
    spirv_cross_construct,
    spirv_cross_destruct,
    spirv_cross_invoke,
};

const struct spirv_cross_interface *spirv_cross_get_interface(void)
{
    return &vtable;
}
