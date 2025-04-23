# Update RuleForm to include category selection
class RuleForm(FlaskForm):
    address_pattern = StringField('Address Pattern', validators=[DataRequired(), Length(max=255)])
    sorting_destination = StringField('Sorting Destination', validators=[DataRequired(), Length(max=255)])
    priority = IntegerField('Priority', validators=[Optional()], default=0)
    category_id = SelectField('Category', coerce=int, validators=[Optional()])
    submit = SubmitField('Save Rule')