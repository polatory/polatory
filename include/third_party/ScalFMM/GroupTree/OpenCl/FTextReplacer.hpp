#ifndef FTEXTREPLACER_HPP
#define FTEXTREPLACER_HPP

#include "../../Utils/FGlobal.hpp"
#include "../../Utils/FAssert.hpp"

#include <sstream>
#include <fstream>
#include <vector>

#include <memory>

#include <sstream>

class FTextReplacer{
protected:
    std::vector<char> content;

    size_t replaceCore(const size_t from, const char* keyStr, const size_t keyLength, const char* valueStr, const size_t valueLength){
        if(keyLength == 0){
            return content.size();
        }

        size_t iter = from;
        while(iter < (content.size()-keyLength+1) && content[iter] != keyStr[0]){
            iter += 1;
        }

        if(iter < (content.size()-keyLength+1)){
            FAssertLF(content[iter] == keyStr[0]);
            size_t counter = 0;
            while(counter < keyLength && content[iter+counter] == keyStr[counter]){
                counter += 1;
            }
            if(counter == keyLength){
                // Found
                if(keyLength < valueLength){
                    const size_t shift = valueLength-keyLength;
                    content.resize(content.size() + shift, '\0');
                    for(size_t idxCopy = content.size()-1 ; idxCopy >= iter+shift ; --idxCopy){
                        content[idxCopy] = content[idxCopy-shift];
                    }
                }

                // Copy at pos iter
                for(size_t idxCopy = 0 ; idxCopy < valueLength ; ++idxCopy){
                    content[iter + idxCopy] = valueStr[idxCopy];
                }

                // We shift some values
                if(valueLength <= keyLength){
                    const size_t shift = keyLength-valueLength;
                    for(size_t idxCopy = iter+valueLength ; idxCopy < content.size()-shift ; ++idxCopy){
                        content[idxCopy] = content[idxCopy+shift];
                    }
                    content.resize(content.size()-shift);
                }
            }
            return iter+1;
        }
        return content.size();
    }

public:
    FTextReplacer(const char* inFilename){
        FAssertLF(inFilename);
        FILE* kernelFile = fopen(inFilename, "r");
        FAssertLF(kernelFile, "Cannot open " , inFilename);

        fseek(kernelFile, 0, SEEK_END);
        const size_t kernelFileSize = ftell(kernelFile);

        content.resize(kernelFileSize+1);
        content[kernelFileSize] = '\0';

        rewind(kernelFile);
        FAssertLF(fread(content.data(), sizeof(char),  kernelFileSize, kernelFile) == kernelFileSize);
        fclose(kernelFile);
    }

    FTextReplacer(const char* inData, const size_t inLength){
        content.resize(inLength+1);
        content[inLength] = '\0';
        memcpy(content.data(), inData, inLength);
    }

    size_t getLength() const{
        return content.size();
    }

    const char* getContent() const{
        return content.data();
    }

    template <class ValueClass>
    int replaceAll(const char* keyStr, const ValueClass& value){
        std::ostringstream stream;
        stream << value;
        const std::string streamStr = stream.str();
        const char* valueStr = streamStr.c_str();
        const size_t valueLength = streamStr.size();

        const size_t keyLength = strlen(keyStr);

        int counterOccurence = 0;
        size_t from = 0;
        while((from = replaceCore(from, keyStr, keyLength, valueStr, valueLength)) != content.size()){
            counterOccurence += 1;
        }

        return counterOccurence;
    }

    template <class ValueClass>
    bool replaceOne(const char* keyStr, const ValueClass& value){
        std::ostringstream stream;
        stream << value;
        const std::string streamStr = stream.str();
        const char* valueStr = streamStr.c_str();
        const size_t valueLength = streamStr.size();

        return replaceCore(0, keyStr, strlen(keyStr), valueStr, valueLength) != content.size();
    }

    void clear(){
        content.clear();
    }
};

#endif // FTEXTREPLACER_HPP

