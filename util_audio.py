import json
import platform

# Platform-specific imports
if platform.system() == "Windows":
    import pyaudiowpatch as pyaudio
else:
    import pyaudio

# Standard list of sample rates.
STANDARD_SAMPLE_RATES = [
    8000.0, 9600.0, 11025.0, 12000.0,
    16000.0, 22050.0, 24000.0, 32000.0,
    44100.0, 48000.0, 88200.0, 96000.0,
    192000.0
]

def compute_supported_sample_rates(p, devinfo, standard_sample_rates=None):
    """
    Computes and returns the supported sample rates for a device.
    
    It tests for:
      - Input supported rates (if the device supports input)
      - Output supported rates (if the device supports output)
      - Full duplex rates (if the device supports both)
    
    Args:
        p: A valid PyAudio instance.
        devinfo: Device information dictionary.
        standard_sample_rates: List of sample rates to test.
        
    Returns:
        A dictionary with keys:
          'input_supported_rates', 'output_supported_rates', 'full_duplex_rates'
    """
    if standard_sample_rates is None:
        standard_sample_rates = STANDARD_SAMPLE_RATES

    input_supported_rates = []
    output_supported_rates = []
    full_duplex_rates = []
    
    for rate in standard_sample_rates:
        # Check input support.
        if devinfo.get("maxInputChannels", 0) > 0:
            try:
                if p.is_format_supported(
                    rate,
                    input_device=devinfo["index"],
                    input_channels=int(devinfo["maxInputChannels"]),
                    input_format=pyaudio.paInt16
                ):
                    input_supported_rates.append(rate)
            except Exception:
                pass
        
        # Check output support.
        if devinfo.get("maxOutputChannels", 0) > 0:
            try:
                if p.is_format_supported(
                    rate,
                    output_device=devinfo["index"],
                    output_channels=int(devinfo["maxOutputChannels"]),
                    output_format=pyaudio.paInt16
                ):
                    output_supported_rates.append(rate)
            except Exception:
                pass
        
        # Check full duplex support.
        if (devinfo.get("maxInputChannels", 0) > 0 and 
            devinfo.get("maxOutputChannels", 0) > 0):
            try:
                if p.is_format_supported(
                    rate,
                    input_device=devinfo["index"],
                    input_channels=int(devinfo["maxInputChannels"]),
                    input_format=pyaudio.paInt16,
                    output_device=devinfo["index"],
                    output_channels=int(devinfo["maxOutputChannels"]),
                    output_format=pyaudio.paInt16
                ):
                    full_duplex_rates.append(rate)
            except Exception:
                pass
    
    return {
        "input_supported_rates": input_supported_rates,
        "output_supported_rates": output_supported_rates,
        "full_duplex_rates": full_duplex_rates
    }

def get_portaudio_info(p=None):
    """
    Returns PortAudio global information.
    
    If no PyAudio instance is provided, one is created internally.
    """
    created = False
    if p is None:
        p = pyaudio.PyAudio()
        created = True

    result = {
        "version": pyaudio.get_portaudio_version(),
        "version_text": pyaudio.get_portaudio_version_text(),
        "host_api_count": p.get_host_api_count(),
        "device_count": p.get_device_count()
    }
    
    if created:
        p.terminate()
    return result

def get_host_apis(p=None, portaudio_info=None):
    """
    Returns a list of host API information dictionaries.
    
    If no PyAudio instance is provided, one is created internally.
    """
    created = False
    if p is None:
        p = pyaudio.PyAudio()
        created = True

    if portaudio_info is None:
        host_api_count = p.get_host_api_count()
    else:
        host_api_count = portaudio_info["host_api_count"]
    
    host_apis = []
    for i in range(host_api_count):
        apiinfo = p.get_host_api_info_by_index(i)
        host_apis.append(apiinfo)
    
    if created:
        p.terminate()
    return host_apis

def get_default_devices(p=None, standard_sample_rates=None):
    """
    Returns a dictionary with default input and output device info,
    including supported sample rates.
    
    If no PyAudio instance is provided, one is created internally.
    """
    created = False
    if p is None:
        p = pyaudio.PyAudio()
        created = True
        
    if standard_sample_rates is None:
        standard_sample_rates = STANDARD_SAMPLE_RATES
        
    default_devices = {}
    try:
        default_input = p.get_default_input_device_info()
        hostapi_input = p.get_host_api_info_by_index(default_input.get("hostApi"))
        default_input["hostApiName"] = hostapi_input.get("name", "")
        rates = compute_supported_sample_rates(p, default_input, standard_sample_rates)
        default_input.update(rates)
        default_devices["input"] = default_input
    except Exception:
        default_devices["input"] = None

    try:
        default_output = p.get_default_output_device_info()
        hostapi_output = p.get_host_api_info_by_index(default_output.get("hostApi"))
        default_output["hostApiName"] = hostapi_output.get("name", "")
        rates = compute_supported_sample_rates(p, default_output, standard_sample_rates)
        default_output.update(rates)
        default_devices["output"] = default_output
    except Exception:
        default_devices["output"] = None

    if created:
        p.terminate()
    return default_devices

def get_devices_info(p=None, standard_sample_rates=None):
    """
    Returns a dictionary with two keys: 'input' and 'output'.
    Each key maps to a list of device info dictionaries that support input or output,
    respectively. Each device info dictionary is augmented with:
      - 'hostApiName': friendly name of its host API.
      - 'input_supported_rates', 'output_supported_rates', 'full_duplex_rates'.
      
    Devices that support both input and output will appear in both lists.
    
    If no PyAudio instance is provided, one is created internally.
    """
    created = False
    if p is None:
        p = pyaudio.PyAudio()
        created = True
        
    if standard_sample_rates is None:
        standard_sample_rates = STANDARD_SAMPLE_RATES
        
    input_devices = []
    output_devices = []
    device_count = p.get_device_count()
    
    for i in range(device_count):
        devinfo = p.get_device_info_by_index(i)
        try:
            hostapi = p.get_host_api_info_by_index(devinfo.get("hostApi"))
            devinfo["hostApiName"] = hostapi.get("name", "")
        except Exception:
            devinfo["hostApiName"] = ""
        
        rates = compute_supported_sample_rates(p, devinfo, standard_sample_rates)
        devinfo.update(rates)
        
        if devinfo.get("maxInputChannels", 0) > 0:
            input_devices.append(devinfo)
        if devinfo.get("maxOutputChannels", 0) > 0:
            output_devices.append(devinfo)
    
    if created:
        p.terminate()
    return {"input": input_devices, "output": output_devices}

def get_wasapi_info(p=None, standard_sample_rates=None):
    """
    Returns WASAPI-specific information if available, including supported sample rates.
    
    The returned dictionary includes:
      - 'info': WASAPI host API info.
      - 'input_devices': A list of WASAPI devices that support input.
      - 'output_devices': A list of WASAPI devices that support output.
    
    If no PyAudio instance is provided, one is created internally.
    Returns None on macOS since WASAPI is Windows-only.
    """
    # WASAPI is Windows-only
    if platform.system() != "Windows":
        return None
        
    created = False
    if p is None:
        p = pyaudio.PyAudio()
        created = True
        
    if standard_sample_rates is None:
        standard_sample_rates = STANDARD_SAMPLE_RATES
        
    wasapi_data = {}
    try:
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        wasapi_data["info"] = wasapi_info
        wasapi_index = wasapi_info["index"]
        input_devices = []
        output_devices = []
        device_count = p.get_device_count()
        for i in range(device_count):
            devinfo = p.get_device_info_by_index(i)
            if devinfo.get("hostApi") == wasapi_index:
                try:
                    hostapi = p.get_host_api_info_by_index(devinfo.get("hostApi"))
                    devinfo["hostApiName"] = hostapi.get("name", "")
                except Exception:
                    devinfo["hostApiName"] = ""
                rates = compute_supported_sample_rates(p, devinfo, standard_sample_rates)
                devinfo.update(rates)
                if devinfo.get("maxInputChannels", 0) > 0:
                    input_devices.append(devinfo)
                if devinfo.get("maxOutputChannels", 0) > 0:
                    output_devices.append(devinfo)
        wasapi_data["input_devices"] = input_devices
        wasapi_data["output_devices"] = output_devices
    except Exception:
        wasapi_data = None

    if created:
        p.terminate()
    return wasapi_data

def get_wasapi_devices_info(p=None, standard_sample_rates=None):
    """
    Returns only the WASAPI devices information as a dictionary with keys:
      - 'input': list of WASAPI input devices.
      - 'output': list of WASAPI output devices.
      
    If no PyAudio instance is provided, one is created internally.
    """
    wasapi_data = get_wasapi_info(p, standard_sample_rates)
    if wasapi_data is None:
        return {"input": [], "output": []}
    return {
        "input": wasapi_data.get("input_devices", []),
        "output": wasapi_data.get("output_devices", [])
    }

def get_wasapi_input_device_id_by_name(device_name, p=None, standard_sample_rates=None):
    """
    Searches WASAPI input devices for a device whose 'name' matches the provided
    device_name (case-insensitive) and returns its device index.
    
    If no matching device is found, returns None.
    
    Args:
        device_name: The device name string to match.
        p: An instance of PyAudio (optional).
        standard_sample_rates: List of sample rates to test (optional).
    """
    devices_info = get_wasapi_devices_info(p, standard_sample_rates)
    input_devices = devices_info.get("input", [])
    for dev in input_devices:
        if dev.get("name", "").lower() == device_name.lower():
            return dev.get("index")
    return None

def get_wasapi_output_device_id_by_name(device_name, p=None, standard_sample_rates=None):
    """
    Searches WASAPI output devices for a device whose 'name' matches the provided
    device_name (case-insensitive) and returns its device index.
    
    If no matching device is found, returns None.
    
    Args:
        device_name: The device name string to match.
        p: An instance of PyAudio (optional).
        standard_sample_rates: List of sample rates to test (optional).
    """
    devices_info = get_wasapi_devices_info(p, standard_sample_rates)
    output_devices = devices_info.get("output", [])
    for dev in output_devices:
        if dev.get("name", "").lower() == device_name.lower():
            return dev.get("index")
    return None

def get_mme_devices_info(p=None, standard_sample_rates=None):
    """
    Returns only the MME devices information as a dictionary with keys:
      - 'input': list of MME input devices.
      - 'output': list of MME output devices.
    
    This function filters the devices retrieved from get_devices_info() by the host API name "MME".
    
    Args:
        p: An instance of PyAudio (optional).
        standard_sample_rates: List of sample rates to test (optional).
    
    Returns:
        A dictionary with keys 'input' and 'output' containing lists of device info dictionaries.
    """
    devices_info = get_devices_info(p, standard_sample_rates)
    mme_input = [dev for dev in devices_info.get("input", [])
                 if dev.get("hostApiName", "").lower() == "mme"]
    mme_output = [dev for dev in devices_info.get("output", [])
                  if dev.get("hostApiName", "").lower() == "mme"]
    return {"input": mme_input, "output": mme_output}

def get_mme_input_device_id_by_name(device_name, p=None, standard_sample_rates=None):
    """
    Searches MME input devices for a device whose 'name' matches the provided
    device_name (case-insensitive) and returns its device index.

    If no matching device is found, returns None.

    Args:
        device_name: The device name string to match.
        p: An instance of PyAudio (optional).
        standard_sample_rates: List of sample rates to test (optional).
    """
    devices_info = get_devices_info(p, standard_sample_rates)
    input_devices = devices_info.get("input", [])
    for dev in input_devices:
        if dev.get("hostApiName", "").lower() == "mme" and dev.get("name", "").lower() == device_name.lower():
            return dev.get("index")
    return None


def get_mme_output_device_id_by_name(device_name, p=None, standard_sample_rates=None):
    """
    Searches MME output devices for a device whose 'name' matches the provided
    device_name (case-insensitive) and returns its device index.

    If no matching device is found, returns None.

    Args:
        device_name: The device name string to match.
        p: An instance of PyAudio (optional).
        standard_sample_rates: List of sample rates to test (optional).
    """
    devices_info = get_devices_info(p, standard_sample_rates)
    output_devices = devices_info.get("output", [])
    for dev in output_devices:
        if dev.get("hostApiName", "").lower() == "mme" and dev.get("name", "").lower() == device_name.lower():
            return dev.get("index")
    return None

def get_directsound_devices_info(p=None, standard_sample_rates=None):
    """
    Returns only the Windows DirectSound devices information as a dictionary with keys:
      - 'input': list of DirectSound input devices.
      - 'output': list of DirectSound output devices.
    
    This function filters the devices retrieved from get_devices_info() by the host API name "Windows DirectSound".
    
    Args:
        p: An instance of PyAudio (optional).
        standard_sample_rates: List of sample rates to test (optional).
    
    Returns:
        A dictionary with keys 'input' and 'output' containing lists of device info dictionaries.
    """
    devices_info = get_devices_info(p, standard_sample_rates)
    directsound_input = [dev for dev in devices_info.get("input", [])
                           if dev.get("hostApiName", "").lower() == "windows directsound"]
    directsound_output = [dev for dev in devices_info.get("output", [])
                            if dev.get("hostApiName", "").lower() == "windows directsound"]
    return {"input": directsound_input, "output": directsound_output}

def get_directsound_input_device_id_by_name(device_name, p=None, standard_sample_rates=None):
    """
    Searches Windows DirectSound input devices for a device whose 'name' matches the provided
    device_name (case-insensitive) and returns its device index.

    If no matching device is found, returns None.

    Args:
        device_name: The device name string to match.
        p: An instance of PyAudio (optional).
        standard_sample_rates: List of sample rates to test (optional).
    """
    devices_info = get_devices_info(p, standard_sample_rates)
    input_devices = devices_info.get("input", [])
    for dev in input_devices:
        if dev.get("hostApiName", "").lower() == "windows directsound" and dev.get("name", "").lower() == device_name.lower():
            return dev.get("index")
    return None


def get_directsound_output_device_id_by_name(device_name, p=None, standard_sample_rates=None):
    """
    Searches Windows DirectSound output devices for a device whose 'name' matches the provided
    device_name (case-insensitive) and returns its device index.

    If no matching device is found, returns None.

    Args:
        device_name: The device name string to match.
        p: An instance of PyAudio (optional).
        standard_sample_rates: List of sample rates to test (optional).
    """
    devices_info = get_devices_info(p, standard_sample_rates)
    output_devices = devices_info.get("output", [])
    for dev in output_devices:
        if dev.get("hostApiName", "").lower() == "windows directsound" and dev.get("name", "").lower() == device_name.lower():
            return dev.get("index")
    return None

def get_device_info_by_id(device_id, p=None, standard_sample_rates=None):
    """
    Retrieves device information by its index.
    
    The function fetches the device info, adds a friendly host API name, and
    computes the supported sample rates (input, output, and full duplex).
    
    Args:
        device_id: The device index to look up.
        p: An instance of PyAudio (optional). If not provided, one is created internally.
        standard_sample_rates: List of sample rates to test (optional).
    
    Returns:
        A device info dictionary with additional keys for 'hostApiName', 
        'input_supported_rates', 'output_supported_rates', and 'full_duplex_rates',
        or None if the device is not found.
    """
    created = False
    if p is None:
        p = pyaudio.PyAudio()
        created = True
        
    try:
        devinfo = p.get_device_info_by_index(device_id)
        try:
            hostapi = p.get_host_api_info_by_index(devinfo.get("hostApi"))
            devinfo["hostApiName"] = hostapi.get("name", "")
        except Exception:
            devinfo["hostApiName"] = ""
        rates = compute_supported_sample_rates(p, devinfo, standard_sample_rates)
        devinfo.update(rates)
    except Exception:
        devinfo = None

    if created:
        p.terminate()
    return devinfo

def get_wasapi_default_output_loopback_device(p=None):
    """
    Returns the default WASAPI loopback output device info.
    
    If no PyAudio instance is provided, one is created internally.
    Returns None on macOS since WASAPI is Windows-only.
    """
    # WASAPI is Windows-only
    if platform.system() != "Windows":
        return None
        
    created = False
    if p is None:
        p = pyaudio.PyAudio()
        created = True
    
    try:
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_output = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
        if not default_output["isLoopbackDevice"]:
            for loopback in p.get_loopback_device_info_generator():
                if default_output["name"] in loopback["name"]:
                    default_output = loopback
                    break
            else:
                default_output = None
    except Exception:
        default_output = None
    
    if created:
        p.terminate()
    return default_output

def get_wasapi_default_input_device(p=None):
    """
    Returns the default WASAPI input device info.
    
    If no PyAudio instance is provided, one is created internally.
    Returns None on macOS since WASAPI is Windows-only.
    """
    # WASAPI is Windows-only
    if platform.system() != "Windows":
        return None
        
    created = False
    if p is None:
        p = pyaudio.PyAudio()
        created = True
    
    try:
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_input = p.get_device_info_by_index(wasapi_info["defaultInputDevice"])
    except Exception:
        default_input = None
    
    if created:
        p.terminate()
    return default_input

# Cross-platform device helper functions

def get_coreaudio_devices_info(p=None, standard_sample_rates=None):
    """
    Returns only the Core Audio devices information as a dictionary with keys:
      - 'input': list of Core Audio input devices.
      - 'output': list of Core Audio output devices.
    
    This function filters the devices retrieved from get_devices_info() by the host API name "Core Audio".
    Returns empty lists on non-macOS platforms.
    
    Args:
        p: An instance of PyAudio (optional).
        standard_sample_rates: List of sample rates to test (optional).
    
    Returns:
        A dictionary with keys 'input' and 'output' containing lists of device info dictionaries.
    """
    if platform.system() != "Darwin":
        return {"input": [], "output": []}
        
    devices_info = get_devices_info(p, standard_sample_rates)
    coreaudio_input = [dev for dev in devices_info.get("input", [])
                      if dev.get("hostApiName", "").lower() == "core audio"]
    coreaudio_output = [dev for dev in devices_info.get("output", [])
                       if dev.get("hostApiName", "").lower() == "core audio"]
    return {"input": coreaudio_input, "output": coreaudio_output}

def get_coreaudio_input_device_id_by_name(device_name, p=None, standard_sample_rates=None):
    """
    Searches Core Audio input devices for a device whose 'name' matches the provided
    device_name (case-insensitive) and returns its device index.

    If no matching device is found, returns None.
    Returns None on non-macOS platforms.

    Args:
        device_name: The device name string to match.
        p: An instance of PyAudio (optional).
        standard_sample_rates: List of sample rates to test (optional).
    """
    if platform.system() != "Darwin":
        return None
        
    devices_info = get_devices_info(p, standard_sample_rates)
    input_devices = devices_info.get("input", [])
    for dev in input_devices:
        if dev.get("hostApiName", "").lower() == "core audio" and dev.get("name", "").lower() == device_name.lower():
            return dev.get("index")
    return None

def get_coreaudio_output_device_id_by_name(device_name, p=None, standard_sample_rates=None):
    """
    Searches Core Audio output devices for a device whose 'name' matches the provided
    device_name (case-insensitive) and returns its device index.

    If no matching device is found, returns None.
    Returns None on non-macOS platforms.

    Args:
        device_name: The device name string to match.
        p: An instance of PyAudio (optional).
        standard_sample_rates: List of sample rates to test (optional).
    """
    if platform.system() != "Darwin":
        return None
        
    devices_info = get_devices_info(p, standard_sample_rates)
    output_devices = devices_info.get("output", [])
    for dev in output_devices:
        if dev.get("hostApiName", "").lower() == "core audio" and dev.get("name", "").lower() == device_name.lower():
            return dev.get("index")
    return None

def get_default_input_device_cross_platform(p=None):
    """
    Returns the default input device for the current platform.
    
    On Windows: Attempts to get WASAPI default input device first, falls back to system default.
    On macOS: Gets the system default input device.
    On other platforms: Gets the system default input device.
    """
    if platform.system() == "Windows":
        wasapi_device = get_wasapi_default_input_device(p)
        if wasapi_device:
            return wasapi_device
    
    # Fall back to system default
    default_devices = get_default_devices(p)
    return default_devices.get("input")

def get_platform_devices_info(p=None, standard_sample_rates=None):
    """
    Returns device information for the current platform's preferred audio API.
    
    On Windows: Returns WASAPI devices if available, falls back to all devices.
    On macOS: Returns Core Audio devices if available, falls back to all devices.
    On other platforms: Returns all devices.
    """
    if platform.system() == "Windows":
        wasapi_devices = get_wasapi_devices_info(p, standard_sample_rates)
        if wasapi_devices and (wasapi_devices.get("input") or wasapi_devices.get("output")):
            return wasapi_devices
    elif platform.system() == "Darwin":
        coreaudio_devices = get_coreaudio_devices_info(p, standard_sample_rates)
        if coreaudio_devices and (coreaudio_devices.get("input") or coreaudio_devices.get("output")):
            return coreaudio_devices
    
    # Fall back to all devices
    return get_devices_info(p, standard_sample_rates)

def get_all_audio_data(p=None):
    """
    Wraps all individual functions into one function that returns the complete audio data.
    
    The returned dictionary has the following top-level keys:
      - 'portaudio'
      - 'host_apis'
      - 'default_devices'
      - 'devices'  (divided into 'input' and 'output' lists)
      - 'wasapi' (Windows only)
      - 'coreaudio' (macOS only)
      - 'platform_devices' (preferred API for current platform)
    
    If no PyAudio instance is provided, one is created internally.
    """
    created = False
    if p is None:
        p = pyaudio.PyAudio()
        created = True
    
    portaudio = get_portaudio_info(p)
    host_apis = get_host_apis(p, portaudio)
    default_devices = get_default_devices(p, STANDARD_SAMPLE_RATES)
    devices = get_devices_info(p, STANDARD_SAMPLE_RATES)
    wasapi = get_wasapi_info(p, STANDARD_SAMPLE_RATES)
    coreaudio = get_coreaudio_devices_info(p, STANDARD_SAMPLE_RATES)
    platform_devices = get_platform_devices_info(p, STANDARD_SAMPLE_RATES)
    
    total_data = {
        "portaudio": portaudio,
        "host_apis": host_apis,
        "default_devices": default_devices,
        "devices": devices,
        "wasapi": wasapi,
        "coreaudio": coreaudio,
        "platform_devices": platform_devices
    }
    
    if created:
        p.terminate()
    return total_data

# Example usage:
if __name__ == "__main__":
    # Using functions without providing a PyAudio instance.
    
    all_data = get_all_audio_data()
    print("All Audio Data:")
    print(json.dumps(all_data, indent=2))
    
    wasapi_devices = get_wasapi_devices_info()
    print("\nWASAPI Devices Info:")
    print(json.dumps(wasapi_devices, indent=2))
    
    search_input_name = "Your Input Device Name Here"
    input_device_id = get_wasapi_input_device_id_by_name(search_input_name)
    if input_device_id is not None:
        print(f"\nFound WASAPI input device '{search_input_name}' with index: {input_device_id}")
    else:
        print(f"\nNo WASAPI input device found with name '{search_input_name}'.")
    
    search_output_name = "Your Output Device Name Here"
    output_device_id = get_wasapi_output_device_id_by_name(search_output_name)
    if output_device_id is not None:
        print(f"\nFound WASAPI output device '{search_output_name}' with index: {output_device_id}")
    else:
        print(f"\nNo WASAPI output device found with name '{search_output_name}'.")
    
    # Look up a device by its ID.
    device_id = 0  # Replace with the desired device index.
    device_info = get_device_info_by_id(device_id)
    if device_info is not None:
        print(f"\nDevice info for device with index {device_id}:")
        print(json.dumps(device_info, indent=2))
    else:
        print(f"\nNo device found with index {device_id}.")
