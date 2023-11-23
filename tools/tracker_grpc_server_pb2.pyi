from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Image(_message.Message):
    __slots__ = ["width", "height", "num_channels", "data"]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    NUM_CHANNELS_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    width: int
    height: int
    num_channels: int
    data: bytes
    def __init__(self, width: _Optional[int] = ..., height: _Optional[int] = ..., num_channels: _Optional[int] = ..., data: _Optional[bytes] = ...) -> None: ...

class Response(_message.Message):
    __slots__ = ["success", "error_msg"]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_MSG_FIELD_NUMBER: _ClassVar[int]
    success: bool
    error_msg: str
    def __init__(self, success: bool = ..., error_msg: _Optional[str] = ...) -> None: ...

class Void(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class InitialMask(_message.Message):
    __slots__ = ["frame", "mask"]
    FRAME_FIELD_NUMBER: _ClassVar[int]
    MASK_FIELD_NUMBER: _ClassVar[int]
    frame: Image
    mask: Image
    def __init__(self, frame: _Optional[_Union[Image, _Mapping]] = ..., mask: _Optional[_Union[Image, _Mapping]] = ...) -> None: ...

class TrackResponse(_message.Message):
    __slots__ = ["success", "mask", "scores", "error_msg"]
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MASK_FIELD_NUMBER: _ClassVar[int]
    SCORES_FIELD_NUMBER: _ClassVar[int]
    ERROR_MSG_FIELD_NUMBER: _ClassVar[int]
    success: bool
    mask: Image
    scores: Image
    error_msg: str
    def __init__(self, success: bool = ..., mask: _Optional[_Union[Image, _Mapping]] = ..., scores: _Optional[_Union[Image, _Mapping]] = ..., error_msg: _Optional[str] = ...) -> None: ...

class InstanceResponse(_message.Message):
    __slots__ = ["instance_id", "token", "error_msg"]
    INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    TOKEN_FIELD_NUMBER: _ClassVar[int]
    ERROR_MSG_FIELD_NUMBER: _ClassVar[int]
    instance_id: int
    token: int
    error_msg: str
    def __init__(self, instance_id: _Optional[int] = ..., token: _Optional[int] = ..., error_msg: _Optional[str] = ...) -> None: ...

class StatefulInitialMask(_message.Message):
    __slots__ = ["instance_id", "token", "frame", "mask"]
    INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    TOKEN_FIELD_NUMBER: _ClassVar[int]
    FRAME_FIELD_NUMBER: _ClassVar[int]
    MASK_FIELD_NUMBER: _ClassVar[int]
    instance_id: int
    token: int
    frame: Image
    mask: Image
    def __init__(self, instance_id: _Optional[int] = ..., token: _Optional[int] = ..., frame: _Optional[_Union[Image, _Mapping]] = ..., mask: _Optional[_Union[Image, _Mapping]] = ...) -> None: ...

class StatefulTrackRequest(_message.Message):
    __slots__ = ["instance_id", "token", "frame"]
    INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    TOKEN_FIELD_NUMBER: _ClassVar[int]
    FRAME_FIELD_NUMBER: _ClassVar[int]
    instance_id: int
    token: int
    frame: Image
    def __init__(self, instance_id: _Optional[int] = ..., token: _Optional[int] = ..., frame: _Optional[_Union[Image, _Mapping]] = ...) -> None: ...

class TrackerInstance(_message.Message):
    __slots__ = ["instance_id", "token"]
    INSTANCE_ID_FIELD_NUMBER: _ClassVar[int]
    TOKEN_FIELD_NUMBER: _ClassVar[int]
    instance_id: int
    token: int
    def __init__(self, instance_id: _Optional[int] = ..., token: _Optional[int] = ...) -> None: ...
