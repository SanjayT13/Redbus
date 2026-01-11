from pydantic import BaseModel, Field,field_validator,ValidationInfo
from typing import List, Annotated
import re 
from datetime import datetime 

Postive_Id = Annotated[int, Field(ge=1,le = 100 )]

class input_schema(BaseModel):
    route_key : List[str] = Field(..., description="Route key in the format 'YYYY-MM-DD_srcid_destid'")
    doj : List[str] = Field(..., description="Date of journey in the format 'YYYY-MM-DD'")
    srcid : List[Postive_Id] = Field(..., description="Source station ID")
    destid : List[Postive_Id] = Field(..., description="Destination station ID") 

    # lenth consistency validator
    @field_validator("doj") 
    def validate_list_length(cls,doj_values,info: ValidationInfo): 
        """
        Ensure that all list inputs have the same length
        """
        expected_length = len(doj_values) 
        for field_name, field_value in info.data.items():
            if field_value is None : 
                continue 
            if len(field_value) != expected_length:
                raise ValueError(f"All input lists must have the same length. Mismatch found in field '{field_name}'.")
        return doj_values

    # date validator
    @field_validator("doj", mode="before")
    def validate_doj_format(cls, doj_values):
        """
        Validate that each date in 'doj' follows the 'YYYY-MM-DD' format
        """
        date_format = "%Y-%m-%d" 
        for d in doj_values:
            try:
                datetime.strptime(d, date_format)
            except ValueError:
                raise ValueError(f"Date '{d}' is not in the format 'YYYY-MM-DD'.")
        return doj_values
    # postive integer validator 
    @field_validator("srcid","destid",mode="before")
    def validate_positive_integers(cls, int_values,field):
        """
        Validate that each integer in 'srcid' and 'destid' is positive
        """
        for val in int_values:
            if val < 0:
                raise ValueError(f"Value '{val}' in field '{field.name}' is not a positive integer.")
        return int_values